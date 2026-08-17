from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from config import (
    SIMILAR_THRESHOLD, AUTO_COLLISION_THRESHOLD, PHASH_DIRECT_THRESHOLD,
    FAST_CANDIDATE_LIMIT, DEEP_CANDIDATE_LIMIT, GEOMETRY_CANDIDATE_LIMIT,
    FULL_LIGHT_SCAN_LIMIT, IMAGE_DIR, SSCD_TOP_K,
)
from core.database import get_conn, now_iso
from core.image_features import (
    extract_features, extract_light_features, enrich_deep_features, hash_file_pair,
    hamming_score, pack_f32, unpack_f32, pack_orb, unpack_orb,
    build_signatures, local_features, read_image, prepared_gray,
    pack_local_feature, unpack_local_feature,
)
from core.object_storage import is_object_key, object_key_path
from ai.embedding import cosine_score
from core.copy_index import save_copy_features, load_copy_features, copy_candidate_scores, pack_copy_feature, unpack_copy_feature

log = logging.getLogger(__name__)
_SAVE_LOCK = threading.Lock()


def _hist_score(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if not len(a) or not len(b): return 0.0
    corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return max(0.0, min(100.0, (corr + 1.0) * 50.0))


def _orb_score(a, b):
    if a is None or b is None or len(a) < 5 or len(b) < 5: return 0.0
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = bf.knnMatch(a, b, k=2)
        good = sum(1 for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
        denom = max(12, min(len(a), len(b)))
        return min(100.0, good / denom * 160.0)
    except Exception:
        return 0.0


def _stored_path(file_path: str | None) -> Path | None:
    if not file_path: return None
    try:
        p = object_key_path(file_path) if is_object_key(file_path) else Path(file_path)
        return p if p.exists() and p.is_file() else None
    except Exception:
        return None


def _save_signatures(conn, image_id: int, signatures):
    for s in signatures or []:
        bands = list(s.get("bands") or [""] * 8)[:8]
        bands += [""] * (8 - len(bands))
        conn.execute(
            """INSERT INTO image_signatures(image_id,kind,phash,dhash,b0,b1,b2,b3,b4,b5,b6,b7,created_time)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(image_id,kind) DO UPDATE SET phash=excluded.phash,dhash=excluded.dhash,
               b0=excluded.b0,b1=excluded.b1,b2=excluded.b2,b3=excluded.b3,b4=excluded.b4,b5=excluded.b5,b6=excluded.b6,b7=excluded.b7""",
            (image_id, s.get("kind","full"), s.get("phash",""), s.get("dhash",""), *bands, now_iso()),
        )


def _signature_candidate_ids(conn, signatures):
    clauses=[]; params=[]
    for sig in signatures or []:
        for i,v in enumerate(sig.get("bands") or []):
            if v:
                clauses.append(f"b{i}=?"); params.append(v)
    if not clauses: return set()
    sql = "SELECT DISTINCT image_id FROM image_signatures WHERE " + " OR ".join(clauses) + " LIMIT ?"
    return {int(r[0]) for r in conn.execute(sql, params + [FAST_CANDIDATE_LIMIT]).fetchall()}


def _legacy_band_candidate_ids(conn, f):
    clauses=[]; params=[]
    for i,v in enumerate(f.get("p_bands") or []):
        if v:
            clauses.append(f"p{i}=?"); params.append(v)
    if not clauses: return set()
    rows=conn.execute("SELECT id FROM images WHERE "+" OR ".join(clauses)+" LIMIT ?", params+[FAST_CANDIDATE_LIMIT]).fetchall()
    return {int(r[0]) for r in rows}


def _ai_candidate_ids(conn, f):
    clauses=[]; params=[]
    for i,v in enumerate(f.get("a_bands") or []):
        if v:
            clauses.append(f"a{i}=?"); params.append(v)
    if not clauses: return set()
    rows=conn.execute("SELECT id FROM images WHERE "+" OR ".join(clauses)+" LIMIT ?", params+[FAST_CANDIDATE_LIMIT]).fetchall()
    return {int(r[0]) for r in rows}


def _excluded_ids(conn, query_sha, query_phash):
    rows=conn.execute("SELECT matched_image_id,query_phash FROM false_positive_pairs WHERE query_sha256=? OR query_phash=?", (query_sha, query_phash)).fetchall()
    out=set()
    for r in rows:
        if r["query_phash"] == query_phash or hamming_score(query_phash, r["query_phash"]) >= 98.0:
            out.add(int(r["matched_image_id"]))
    return out


def _quick_scores(f, rows, sig_map):
    qs=f.get("signatures") or [{"phash":f.get("phash"),"dhash":f.get("dhash")}]
    items=[]
    for row in rows:
        stored=sig_map.get(int(row["id"])) or [{"phash":row["phash"],"dhash":row["dhash"]}]
        bestp=bestd=0.0
        for q in qs:
            for s in stored:
                p=hamming_score(q.get("phash"),s.get("phash")); d=hamming_score(q.get("dhash"),s.get("dhash"))
                if p+0.25*d > bestp+0.25*bestd: bestp,bestd=p,d
        roi=hamming_score(f.get("roi_phash"), row["roi_phash"])
        rank=0.62*bestp+0.18*bestd+0.20*roi
        items.append({"row":row,"rank":rank,"p":bestp,"d":bestd,"roi":roi})
    items.sort(key=lambda x:x["rank"], reverse=True)
    return items


def _load_signature_map(conn, ids):
    if not ids: return {}
    ph=','.join('?'*len(ids))
    rows=conn.execute(f"SELECT image_id,kind,phash,dhash FROM image_signatures WHERE image_id IN ({ph})", list(ids)).fetchall()
    out={}
    for r in rows: out.setdefault(int(r["image_id"]),[]).append(dict(r))
    return out


def _full_light_rows(conn):
    return conn.execute("SELECT id,phash,dhash,roi_phash,customer_id,ai_hash FROM images WHERE COALESCE(trusted,1)=1 ORDER BY id DESC LIMIT ?", (FULL_LIGHT_SCAN_LIMIT,)).fetchall()


def _fetch_deep(conn, ids):
    if not ids: return {}
    ph=','.join('?'*len(ids))
    rows=conn.execute(f"SELECT id,file_path,customer_id,color_hist,orb_desc,orb_rows,ai_feature FROM images WHERE COALESCE(trusted,1)=1 AND id IN ({ph})", ids).fetchall()
    return {int(r["id"]):r for r in rows}


def _hull_coverage(points, shape):
    if points is None or len(points) < 3 or shape is None:
        return 0.0
    try:
        pts=np.asarray(points,dtype=np.float32).reshape(-1,2)
        hull=cv2.convexHull(pts)
        area=float(abs(cv2.contourArea(hull)))
        h,w=shape[:2]
        return max(0.0,min(1.0,area/max(1.0,float(h*w))))
    except Exception:
        return 0.0


def _homography_sane(H, qshape, cshape):
    if H is None or qshape is None or cshape is None:
        return False
    try:
        if not np.isfinite(H).all():
            return False
        H=np.asarray(H,dtype=np.float64)
        if abs(H[2,2]) < 1e-10:
            return False
        H=H/H[2,2]
        # Reject extremely ill-conditioned transforms commonly produced by
        # repetitive backgrounds (wood slats, fences, text rows, leaves, etc.).
        if np.linalg.cond(H) > 2e5:
            return False
        qh,qw=qshape[:2]; ch,cw=cshape[:2]
        corners=np.float32([[0,0],[qw,0],[qw,qh],[0,qh]]).reshape(-1,1,2)
        proj=cv2.perspectiveTransform(corners,H).reshape(-1,2)
        if not np.isfinite(proj).all():
            return False
        hull=cv2.convexHull(proj.astype(np.float32))
        area=float(abs(cv2.contourArea(hull)))
        cand_area=max(1.0,float(ch*cw))
        ratio=area/cand_area
        # A crop may legitimately occupy a small region, but degenerate
        # near-zero or astronomically large projections are not real copies.
        if ratio < 0.01 or ratio > 25.0:
            return False
        # The projected quadrilateral must remain convex and non-degenerate.
        if len(hull) < 4:
            return False
        return True
    except Exception:
        return False


def _match_descriptors(qf, cf, norm, qshape=None, cshape=None):
    qd=qf.get("desc"); cd=cf.get("desc")
    empty={"score":0.0,"inliers":0,"ratio":0.0,"H":None,"mutual":0,
           "coverage_q":0.0,"coverage_c":0.0,"homography_sane":False}
    if qd is None or cd is None or len(qd)<6 or len(cd)<6:
        return empty
    try:
        bf=cv2.BFMatcher(norm)
        # Bidirectional Lowe-ratio matching. Requiring mutual correspondence
        # sharply reduces false geometry on repetitive textures.
        fwd=bf.knnMatch(qd,cd,k=2)
        rev=bf.knnMatch(cd,qd,k=2)
        fgood={(m.queryIdx,m.trainIdx):m for pair in fwd if len(pair)==2
               for m in [pair[0]] if m.distance < 0.72*pair[1].distance}
        rgood={(m.trainIdx,m.queryIdx) for pair in rev if len(pair)==2
               for m in [pair[0]] if m.distance < 0.72*pair[1].distance}
        good=[m for key,m in fgood.items() if key in rgood]
        if len(good)<5:
            out=dict(empty); out["score"]=min(55.0,len(good)*9.0); out["mutual"]=len(good)
            return out
        src=np.float32([qf["kp"][m.queryIdx] for m in good]).reshape(-1,1,2)
        dst=np.float32([cf["kp"][m.trainIdx] for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.5)
        if mask is None:
            out=dict(empty); out["mutual"]=len(good); return out
        keep=mask.ravel().astype(bool)
        inliers=int(keep.sum())
        ratio=inliers/max(1,len(good))
        src_in=src.reshape(-1,2)[keep] if inliers else np.empty((0,2),np.float32)
        dst_in=dst.reshape(-1,2)[keep] if inliers else np.empty((0,2),np.float32)
        cov_q=_hull_coverage(src_in,qshape)
        cov_c=_hull_coverage(dst_in,cshape)
        sane=_homography_sane(H,qshape,cshape)
        min_cov=min(cov_q,cov_c); max_cov=max(cov_q,cov_c)

        # Strong geometry now requires mutually consistent matches spread over
        # a meaningful area in both images, not just many points on one fence,
        # text row, wooden wall, foliage cluster, etc.
        if sane and inliers>=18 and ratio>=0.52 and min_cov>=0.025 and max_cov>=0.10:
            score=min(98.8,90.5+min(6.8,(inliers-18)*0.22)+ratio*1.5+min_cov*8)
        elif sane and inliers>=12 and ratio>=0.45 and min_cov>=0.018 and max_cov>=0.075:
            score=min(89.5,78+inliers*0.55+ratio*4+min_cov*10)
        elif sane and inliers>=7 and ratio>=0.35 and min_cov>=0.01:
            score=min(83.0,62+inliers*1.4+ratio*7+min_cov*10)
        else:
            score=min(74.0,len(good)*2.2+ratio*16+min_cov*20)
        return {"score":float(score),"inliers":inliers,"ratio":float(ratio),"H":H,
                "mutual":len(good),"coverage_q":float(cov_q),"coverage_c":float(cov_c),
                "homography_sane":bool(sane)}
    except Exception:
        return empty


def _aligned_ssim(qgray, cgray, H):
    if H is None:
        return 0.0, 0.0
    try:
        h,w=cgray.shape[:2]
        warped=cv2.warpPerspective(qgray,H,(w,h),flags=cv2.INTER_LINEAR)
        mask=cv2.warpPerspective(np.ones(qgray.shape[:2],dtype=np.uint8)*255,H,(w,h),flags=cv2.INTER_NEAREST)>0
        valid=mask & (warped>0)
        count=int(valid.sum())
        overlap=count/max(1,min(qgray.size,cgray.size))
        if count<400 or overlap<0.05:
            return 0.0,float(overlap)
        x=warped[valid].astype(np.float32); y=cgray[valid].astype(np.float32)
        ux=float(x.mean()); uy=float(y.mean())
        vx=float(x.var()); vy=float(y.var()); cov=float(((x-ux)*(y-uy)).mean())
        c1=(0.01*255)**2; c2=(0.03*255)**2
        s=((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2)+1e-8)
        return max(0.0,min(100.0,s*100.0)),float(overlap)
    except Exception:
        return 0.0,0.0


def _geometry_once(query_img, cand_img, query_pre=None, candidate_local=None):
    if query_pre is None:
        qlocal=local_features(query_img); qg=prepared_gray(query_img)
    else:
        qlocal=query_pre["local"]; qg=query_pre["gray"]
    clocal=candidate_local or local_features(cand_img)
    sift=_match_descriptors(qlocal["sift"],clocal["sift"],cv2.NORM_L2,qg.shape,cand_img.shape[:2])
    ak=_match_descriptors(qlocal["akaze"],clocal["akaze"],cv2.NORM_HAMMING,qg.shape,cand_img.shape[:2])
    best=sift if sift["score"]>=ak["score"] else ak
    best["method"]="SIFT" if best is sift else "AKAZE"
    cg=prepared_gray(cand_img)
    ssim,overlap=_aligned_ssim(qg,cg,best.get("H"))
    best["ssim"]=ssim; best["overlap"]=min(1.0,overlap)
    if best["inliers"]>=8 and best["ratio"]>=0.35 and overlap>=0.08:
        best["score"]=max(best["score"], min(99.4, 0.72*best["score"]+0.28*ssim))
    return best


def _load_or_build_candidate_local(conn, image_id: int, cand_img):
    """Persist candidate-side SIFT/AKAZE so repeated checks do not recompute them."""
    try:
        row=conn.execute("SELECT * FROM image_local_features WHERE image_id=?",(int(image_id),)).fetchone()
        if row and row["sift_desc"] and row["akaze_desc"]:
            return {
                "sift": unpack_local_feature(row["sift_kp"],row["sift_desc"],row["sift_rows"],row["sift_cols"],row["sift_dtype"]),
                "akaze": unpack_local_feature(row["akaze_kp"],row["akaze_desc"],row["akaze_rows"],row["akaze_cols"],row["akaze_dtype"]),
            }
    except Exception:
        pass
    local=local_features(cand_img)
    try:
        skp,sdesc,srows,scols,sdtype=pack_local_feature(local["sift"])
        akp,akdesc,akrows,akcols,akdtype=pack_local_feature(local["akaze"])
        conn.execute(
            """INSERT INTO image_local_features(image_id,sift_kp,sift_desc,sift_rows,sift_cols,sift_dtype,akaze_kp,akaze_desc,akaze_rows,akaze_cols,akaze_dtype,updated_time)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(image_id) DO UPDATE SET
                 sift_kp=excluded.sift_kp,sift_desc=excluded.sift_desc,sift_rows=excluded.sift_rows,sift_cols=excluded.sift_cols,sift_dtype=excluded.sift_dtype,
                 akaze_kp=excluded.akaze_kp,akaze_desc=excluded.akaze_desc,akaze_rows=excluded.akaze_rows,akaze_cols=excluded.akaze_cols,akaze_dtype=excluded.akaze_dtype,updated_time=excluded.updated_time""",
            (int(image_id),skp,sdesc,srows,scols,sdtype,akp,akdesc,akrows,akcols,akdtype,now_iso()),
        )
        conn.commit()
    except Exception:
        log.debug("缓存局部特征失败 image=%s",image_id,exc_info=True)
    return local


def _geometry_score(query_img, candidate_path: Path, qcache=None, conn=None, image_id=None):
    try:
        cand_img=read_image(candidate_path)
        if qcache is None: qcache={}
        if "normal" not in qcache:
            qcache["normal"]={"local":local_features(query_img),"gray":prepared_gray(query_img)}
        candidate_local=_load_or_build_candidate_local(conn,int(image_id),cand_img) if conn is not None and image_id is not None else local_features(cand_img)
        normal=_geometry_once(query_img,cand_img,qcache["normal"],candidate_local)
        if normal.get("score",0)>=90:
            return normal
        if "mirror_img" not in qcache:
            qcache["mirror_img"]=cv2.flip(query_img,1)
            qcache["mirror"]={"local":local_features(qcache["mirror_img"]),"gray":prepared_gray(qcache["mirror_img"])}
        mirrored=_geometry_once(qcache["mirror_img"],cand_img,qcache["mirror"],candidate_local)
        if mirrored.get("score",0)>normal.get("score",0):
            mirrored["method"]=(mirrored.get("method") or "")+"+MIRROR"
            return mirrored
        return normal
    except Exception:
        return {"score":0.0,"inliers":0,"ratio":0.0,"H":None,"method":"","ssim":0.0,"overlap":0.0,"mutual":0,"coverage_q":0.0,"coverage_c":0.0,"homography_sane":False}

def _fetch_light_rows_by_ids(conn, ids):
    ids=list({int(x) for x in ids if x is not None})
    if not ids: return []
    out=[]
    for st in range(0,len(ids),700):
        chunk=ids[st:st+700]
        ph=','.join('?'*len(chunk))
        out.extend(conn.execute(
            f"SELECT id,phash,dhash,roi_phash,customer_id,ai_hash FROM images WHERE COALESCE(trusted,1)=1 AND id IN ({ph})",
            chunk,
        ).fetchall())
    return out


def _first_copy_feature(features):
    for item in features.get("copy_views") or []:
        try:
            v=np.asarray(item.get("feature"),dtype=np.float32).reshape(-1)
            if v.size:
                n=float(np.linalg.norm(v))
                return v/n if n>1e-8 else v
        except Exception:
            pass
    return None


def _false_positive_exclusions(conn, query_sha, query_phash, query_copy_vec):
    """Return image/customer exclusions, including transformed variants of a prior false positive."""
    rows=conn.execute(
        """SELECT fp.*, COALESCE(fp.matched_customer_id,i.customer_id) AS cid
           FROM false_positive_pairs fp LEFT JOIN images i ON i.id=fp.matched_image_id"""
    ).fetchall()
    image_ids=set(); customer_ids=set()
    for r in rows:
        same=False
        if r["query_sha256"] and r["query_sha256"]==query_sha:
            same=True
        elif query_phash and r["query_phash"] and hamming_score(query_phash,r["query_phash"])>=96.0:
            same=True
        elif query_copy_vec is not None and r["query_copy_feature"] and r["query_copy_dim"]:
            old=unpack_copy_feature(r["query_copy_feature"],r["query_copy_dim"])
            if old is not None and old.size==query_copy_vec.size:
                sim=float(np.dot(old,query_copy_vec)/(max(1e-8,np.linalg.norm(old))*max(1e-8,np.linalg.norm(query_copy_vec))))*100.0
                if sim>=96.0:
                    same=True
        if same:
            image_ids.add(int(r["matched_image_id"]))
            if r["cid"]:
                customer_ids.add(int(r["cid"]))
    return image_ids,customer_ids


def _ensure_candidate_copy_score(conn, query_views, image_id, candidate_path, existing_score=0.0):
    """Build missing SSCD descriptors lazily for old V1.8 images and return exact max cosine."""
    if not query_views or existing_score>0:
        return float(existing_score)
    try:
        stored=load_copy_features(conn,{int(image_id)}).get(int(image_id),[])
        if not stored and candidate_path:
            from ai.copy_embedding import extract_copy_features
            img=read_image(candidate_path)
            stored=extract_copy_features(img)
            if stored:
                save_copy_features(conn,int(image_id),stored)
                conn.commit()
        if not stored:
            return 0.0
        from ai.copy_embedding import max_cosine
        score,_,_=max_cosine(query_views,stored)
        return float(score)
    except Exception:
        log.debug("候选SSCD特征补建失败 image=%s",image_id,exc_info=True)
        return float(existing_score)


def check_image(path):
    """V1.9: exact hash -> safe light recall -> SSCD copy-AI recall -> geometry verification."""
    sha256,md5=hash_file_pair(path)
    conn=get_conn()
    try:
        row=conn.execute("SELECT id FROM images WHERE COALESCE(trusted,1)=1 AND (sha256=? OR md5=?) LIMIT 1",(sha256,md5)).fetchone()
        if row:
            return {
                "type":"same","score":100.0,"matched_image_id":row["id"],
                "features":{"sha256":sha256,"md5":md5},"match_type":"同一图片",
                "strong_match":True,"learn_safe":False,
            }

        f,img=extract_light_features(path,sha256,md5)
        quality=f.get("quality") or {}
        low_info=bool(quality.get("low_information"))

        candidate_ids=_signature_candidate_ids(conn,f["signatures"]) | _legacy_band_candidate_ids(conn,f)
        all_rows=_full_light_rows(conn)
        indexed_rows=[r for r in all_rows if int(r["id"]) in candidate_ids]
        indexed_scores=_quick_scores(
            f,indexed_rows,_load_signature_map(conn,{int(r["id"]) for r in indexed_rows})
        ) if indexed_rows else []
        full_scores=_quick_scores(f,all_rows,{})[:FAST_CANDIDATE_LIMIT]
        merged={}
        for item in indexed_scores+full_scores:
            iid=int(item["row"]["id"])
            if iid not in merged or item["rank"]>merged[iid]["rank"]:
                merged[iid]=item
        light=sorted(merged.values(),key=lambda x:x["rank"],reverse=True)

        # Respect prior human false-positive decisions even on the fast hash path.
        pre_ex_images,pre_ex_customers=_false_positive_exclusions(conn,sha256,f.get("phash"),None)
        light=[x for x in light if int(x["row"]["id"]) not in pre_ex_images and (not x["row"]["customer_id"] or int(x["row"]["customer_id"]) not in pre_ex_customers)]

        # V1.9.1 SAFE: perceptual hashes are recall signals only. They never
        # auto-confirm a non-exact collision before SSCD/geometry corroboration.

        # Deep features now include both legacy MobileNet auxiliary features and
        # task-specific SSCD copy descriptors. SSCD is an independent candidate
        # recall path, so crop/screenshot changes do not need to survive pHash first.
        enrich_deep_features(f,img)
        copy_ranked=copy_candidate_scores(conn,f.get("copy_views") or [],SSCD_TOP_K)
        copy_score_map={int(x["image_id"]):float(x["score"]) for x in copy_ranked}
        copy_ids=set(copy_score_map)

        ai_ids=_ai_candidate_ids(conn,f)
        have={int(x["row"]["id"]) for x in light}
        extra_ids=(copy_ids | ai_ids) - have
        if extra_ids:
            add_rows=_fetch_light_rows_by_ids(conn,extra_ids)
            if add_rows:
                add_scores=_quick_scores(f,add_rows,_load_signature_map(conn,{int(r["id"]) for r in add_rows}))
                light.extend(add_scores)

        # Cluster-level false-positive memory. A re-crop of a previously rejected
        # pair excludes every alias belonging to that matched customer.
        query_copy=_first_copy_feature(f)
        excluded_images,excluded_customers=_false_positive_exclusions(conn,sha256,f.get("phash"),query_copy)
        light=[x for x in light if int(x["row"]["id"]) not in excluded_images and (not x["row"]["customer_id"] or int(x["row"]["customer_id"]) not in excluded_customers)]
        light.sort(key=lambda x:max(x["rank"],copy_score_map.get(int(x["row"]["id"]),0.0)),reverse=True)

        top_candidates=light[:DEEP_CANDIDATE_LIMIT]
        if not top_candidates:
            return {"type":"new","score":0.0,"features":f}

        deep_map=_fetch_deep(conn,[int(x["row"]["id"]) for x in top_candidates])
        evaluated=[]
        for item in top_candidates:
            iid=int(item["row"]["id"])
            deep=deep_map.get(iid)
            if not deep: continue
            orb=_orb_score(f.get("orb"),unpack_orb(deep["orb_desc"],deep["orb_rows"]))
            ai=cosine_score(f.get("ai_feature"),unpack_f32(deep["ai_feature"]))
            hist=_hist_score(f.get("color_hist"),unpack_f32(deep["color_hist"]))
            copy_ai=copy_score_map.get(iid,0.0)
            base=0.24*item["p"]+0.08*item["d"]+0.10*item["roi"]+0.07*hist+0.15*orb+0.06*ai+0.30*copy_ai
            evaluated.append({
                **item,"deep":deep,"orb":orb,"ai":ai,"copy_ai":copy_ai,"hist":hist,
                "base":base,"geometry":0.0,"inliers":0,"geom_ratio":0.0,"geom_method":"",
                "ssim":0.0,"overlap":0.0,"mutual":0,"coverage_q":0.0,"coverage_c":0.0,"homography_sane":False,
            })
        evaluated.sort(key=lambda x:max(x["base"],x["rank"],x["copy_ai"]),reverse=True)

        # Pairwise verification. Candidate local descriptors are cached in SQLite,
        # so repeated checks become cheaper instead of recomputing SIFT/AKAZE.
        _qgeom_cache={}
        for item in evaluated[:GEOMETRY_CANDIDATE_LIMIT]:
            iid=int(item["row"]["id"])
            cp=_stored_path(item["deep"]["file_path"])
            if cp:
                if item["copy_ai"]<=0 and f.get("copy_views"):
                    item["copy_ai"]=_ensure_candidate_copy_score(conn,f.get("copy_views"),iid,cp,item["copy_ai"])
                g=_geometry_score(img,cp,_qgeom_cache,conn=conn,image_id=iid)
                item["geometry"]=g["score"]; item["inliers"]=g["inliers"]; item["geom_ratio"]=g["ratio"]
                item["geom_method"]=g.get("method",""); item["ssim"]=g.get("ssim",0.0); item["overlap"]=g.get("overlap",0.0)
                item["mutual"]=g.get("mutual",0); item["coverage_q"]=g.get("coverage_q",0.0); item["coverage_c"]=g.get("coverage_c",0.0); item["homography_sane"]=bool(g.get("homography_sane",False))

        best=None
        for item in evaluated:
            independent=sum([
                item["p"]>=92, item["d"]>=90, item["roi"]>=90,
                item["orb"]>=75, item["geometry"]>=84, item["copy_ai"]>=82,
            ])
            cov_min=min(float(item.get("coverage_q") or 0.0),float(item.get("coverage_c") or 0.0))
            cov_max=max(float(item.get("coverage_q") or 0.0),float(item.get("coverage_c") or 0.0))
            geom_structural=bool(
                item["geometry"]>=90 and item["inliers"]>=16 and item["geom_ratio"]>=0.50
                and item.get("homography_sane") and cov_min>=0.025 and cov_max>=0.10
            )
            # Geometry alone is not enough for AUTO90. It must agree with at least
            # one independent copy/structure signal; this prevents repetitive
            # backgrounds from becoming 99% false collisions.
            strong_geometry=bool(geom_structural and (
                item["copy_ai"]>=68 or item["p"]>=88 or item["roi"]>=88 or item["ssim"]>=35
            ))
            # Hashes are recall signals. Auto-confirm from hashes requires SSCD or
            # strong local structure as corroboration.
            strong_hash=bool(not low_info and item["p"]>=98 and max(item["d"],item["roi"])>=95
                             and item["hist"]>=82 and (item["copy_ai"]>=72 or item["orb"]>=72 or item["geometry"]>=82))
            strong_combo=bool(not low_info and item["p"]>=92 and item["orb"]>=72
                              and (item["copy_ai"]>=66 or item["geometry"]>=72) and independent>=3)
            # Task-specific copy AI is powerful, but semantic look-alikes must not
            # auto-cross 90 without independent visual structure.
            strong_copy=bool(not low_info and item["copy_ai"]>=84 and (
                item["geometry"]>=72 or item["p"]>=84 or item["roi"]>=84 or item["orb"]>=66
            ))
            very_strong_copy=bool(not low_info and item["copy_ai"]>=94 and (
                item["p"]>=78 or item["roi"]>=78 or item["orb"]>=55 or item["geometry"]>=60 or item["hist"]>=90
            ))
            # Simple/flat images may legitimately be the same copy, but hashes alone
            # are unsafe. Require colour agreement + SSCD as additional evidence.
            low_info_safe=bool(low_info and item["p"]>=99 and item["d"]>=97 and item["hist"]>=98 and item["copy_ai"]>=94 and item["geometry"]>=70)
            strong=bool(strong_geometry or strong_hash or strong_combo or strong_copy or very_strong_copy or low_info_safe)

            score=max(item["base"],item["geometry"])
            if strong_geometry:
                score=max(score,item["geometry"])
            if strong_hash:
                score=max(score,90+(item["p"]-PHASH_DIRECT_THRESHOLD)*1.5)
            if strong_copy:
                score=max(score,90+min(9.3,max(0.0,item["copy_ai"]-82)*0.62+max(0.0,item["geometry"]-76)*0.08))
            elif very_strong_copy:
                score=max(score,92+min(7.0,(item["copy_ai"]-91)*0.75))
            elif item["copy_ai"]>=70:
                # Useful for candidate ranking/suspect alerts, but never auto-cross 90.
                score=max(score,min(AUTO_COLLISION_THRESHOLD-0.25,82+(item["copy_ai"]-70)*0.48))
            if low_info_safe:
                score=max(score,92+min(5.0,(item["hist"]-97)*0.6+(item["copy_ai"]-90)*0.25))

            if low_info and not (strong_geometry or low_info_safe):
                # Flat/simple/default-avatar images require manual confirmation unless
                # exact file hash or real local geometry proves they are the same copy.
                strong=False
            if not strong:
                score=min(score,AUTO_COLLISION_THRESHOLD-0.25)
            score=min(99.6,max(0.0,score))

            # Alias learning is stricter than auto-collision. This prevents one bad
            # auto verdict from contaminating the customer's image cluster.
            learn_safe=bool(
                (strong_geometry and item["copy_ai"]>=72 and (item["ssim"]>=30 or item["p"]>=90))
                or (not low_info and item["copy_ai"]>=90 and item["geometry"]>=82 and item["inliers"]>=12)
                or (strong_hash and item["copy_ai"]>=80)
            )
            cur={**item,"score":score,"strong":strong,"learn_safe":learn_safe,"independent":independent}
            if best is None or cur["score"]>best["score"]:
                best=cur

        if best and best["score"]>=SIMILAR_THRESHOLD:
            if best["copy_ai"]>=82 and best["copy_ai"]>=max(best["geometry"],best["p"]):
                mtype="AI同图识别"
            elif best["geometry"]>=84:
                mtype="局部同图匹配"
            elif best["p"]>=92:
                mtype="高度相似图片"
            else:
                mtype="视觉相似图片"
            return {
                "type":"similar","score":round(best["score"],2),"matched_image_id":best["row"]["id"],
                "features":f,"match_type":mtype,"strong_match":best["strong"],"learn_safe":best["learn_safe"],
                "detail":{
                    "phash":round(best["p"],1),"dhash":round(best["d"],1),"orb":round(best["orb"],1),
                    "ai":round(best["ai"],1),"copy_ai":round(best["copy_ai"],1),"geometry":round(best["geometry"],1),
                    "inliers":best["inliers"],"method":best["geom_method"],"ssim":round(best.get("ssim",0.0),1),
                    "overlap":round(best.get("overlap",0.0),3),"mutual":int(best.get("mutual",0)),
                    "coverage_q":round(float(best.get("coverage_q",0.0)),4),"coverage_c":round(float(best.get("coverage_c",0.0)),4),
                    "homography_sane":bool(best.get("homography_sane",False)),"low_information":low_info,
                    "blur":round(float(quality.get("lap_var") or 0.0),2),
                },
            }
        return {"type":"new","score":0.0,"features":f}
    finally:
        conn.close()


def _insert_image(conn, customer_id, path, file_id, file_unique_id, submitter, submitter_id, chat_id, source, object_key, f):
    orb_blob,orb_rows=pack_orb(f.get("orb"))
    stored_path=object_key if object_key else str(path)
    vals=[file_id,file_unique_id,stored_path,f["sha256"],f["md5"],f["phash"],f["dhash"],f["roi_phash"],*f["p_bands"],pack_f32(f["color_hist"]),orb_blob,orb_rows,pack_f32(f.get("ai_feature")),f.get("ai_hash",""),*f.get("a_bands",[""]*8),customer_id,submitter,submitter_id,chat_id,source,now_iso()]
    q="""INSERT INTO images(file_id,file_unique_id,file_path,sha256,md5,phash,dhash,roi_phash,p0,p1,p2,p3,p4,p5,p6,p7,color_hist,orb_desc,orb_rows,ai_feature,ai_hash,a0,a1,a2,a3,a4,a5,a6,a7,customer_id,submitter,submitter_id,chat_id,source,created_time) VALUES("""+",".join(["?"]*35)+")"
    cur=conn.execute(q,vals); image_id=cur.lastrowid
    _save_signatures(conn,image_id,f.get("signatures"))
    try:
        save_copy_features(conn,image_id,f.get("copy_views"))
    except Exception:
        log.debug("保存SSCD同图特征失败 image=%s",image_id,exc_info=True)
    if f.get("local"):
        try:
            skp,sdesc,srows,scols,sdtype=pack_local_feature(f["local"]["sift"])
            akp,akdesc,akrows,akcols,akdtype=pack_local_feature(f["local"]["akaze"])
            conn.execute(
                """INSERT OR REPLACE INTO image_local_features(image_id,sift_kp,sift_desc,sift_rows,sift_cols,sift_dtype,akaze_kp,akaze_desc,akaze_rows,akaze_cols,akaze_dtype,updated_time)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (image_id,skp,sdesc,srows,scols,sdtype,akp,akdesc,akrows,akcols,akdtype,now_iso()),
            )
        except Exception:
            log.debug("保存局部特征失败 image=%s",image_id,exc_info=True)
    return image_id


def save_customer_and_image(path,file_id,file_unique_id,submitter,submitter_id,chat_id,customer_data,raw_text,source="live",source_message_id="",object_key=None,features=None):
    # Serialize formal customer creation. This closes the race where two staff
    # submit the same new photo at nearly the same time and both saw "new".
    with _SAVE_LOCK:
        # Handler has already run the full detector and passes its features.
        # Re-running SSCD/SIFT here would double latency. Only callers that did
        # not provide prechecked features need a second full duplicate check.
        if source == "live" and features is None:
            latest = check_image(path)
            if latest.get("type") == "same":
                conn=get_conn(); row=conn.execute("SELECT id,customer_id FROM images WHERE id=?",(latest["matched_image_id"],)).fetchone(); conn.close()
                if row: return row["customer_id"],row["id"],False
            if latest.get("type") == "similar" and latest.get("strong_match") and float(latest.get("score") or 0)>=AUTO_COLLISION_THRESHOLD:
                return save_image_alias(path,latest["matched_image_id"],file_id,file_unique_id,submitter,submitter_id,chat_id,"live_alias",object_key,latest.get("features"))

        f=features or extract_features(path)
        if f.get("orb") is None or (not f.get("copy_views") and f.get("ai_feature") is None): f=extract_features(path)
        conn=get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            exists=conn.execute("SELECT id,customer_id FROM images WHERE COALESCE(trusted,1)=1 AND (sha256=? OR md5=?) LIMIT 1",(f["sha256"],f["md5"])).fetchone()
            if exists:
                conn.rollback(); return exists["customer_id"],exists["id"],False
            cur=conn.execute("""INSERT INTO customers(name,age,job,income,work_year,software,receiver,raw_text,submitter,submitter_id,chat_id,source,source_message_id,created_time) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(customer_data.get("name",""),customer_data.get("age",""),customer_data.get("job",""),customer_data.get("income",""),customer_data.get("work_year",""),customer_data.get("software",""),customer_data.get("receiver",""),raw_text or "",submitter,submitter_id,chat_id,source,str(source_message_id or ""),now_iso()))
            cid=cur.lastrowid
            iid=_insert_image(conn,cid,path,file_id,file_unique_id,submitter,submitter_id,chat_id,source,object_key,f)
            conn.commit(); return cid,iid,True
        except Exception:
            conn.rollback(); raise
        finally: conn.close()

def save_image_alias(path,matched_image_id,file_id="",file_unique_id="",submitter="",submitter_id="",chat_id="",source="alias",object_key=None,features=None):
    f=features or extract_features(path)
    if f.get("orb") is None or (not f.get("copy_views") and f.get("ai_feature") is None): f=extract_features(path)
    conn=get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        exists=conn.execute("SELECT id,customer_id FROM images WHERE COALESCE(trusted,1)=1 AND (sha256=? OR md5=?) LIMIT 1",(f["sha256"],f["md5"])).fetchone()
        if exists: conn.rollback(); return exists["customer_id"],exists["id"],False
        base=conn.execute("SELECT customer_id FROM images WHERE id=? AND COALESCE(trusted,1)=1",(int(matched_image_id),)).fetchone()
        if not base or not base["customer_id"]: conn.rollback(); return None,None,False
        iid=_insert_image(conn,base["customer_id"],path,file_id,file_unique_id,submitter,submitter_id,chat_id,source,object_key,f)
        conn.execute(
            "UPDATE images SET trusted=1,parent_image_id=?,match_evidence=? WHERE id=?",
            (int(matched_image_id), str(source or "alias"), int(iid)),
        )
        conn.commit(); return base["customer_id"],iid,True
    except Exception:
        conn.rollback(); raise
    finally: conn.close()


def create_collision(query_sha256,matched_image_id,submitter,submitter_id,chat_id,match_type,score,query_phash="",query_file_path="",query_file_id="",query_file_unique_id="",query_copy_feature=None):
    blob,dim=pack_copy_feature(query_copy_feature)
    conn=get_conn()
    try:
        cur=conn.execute(
            """INSERT INTO collision_records(query_sha256,query_phash,query_file_path,query_file_id,query_file_unique_id,query_copy_feature,query_copy_dim,matched_image_id,query_submitter,query_submitter_id,chat_id,match_type,score,status,created_time)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)""",
            (query_sha256,query_phash,query_file_path,query_file_id,query_file_unique_id,blob,dim,matched_image_id,submitter,submitter_id,chat_id,match_type,float(score),now_iso()),
        )
        cid=cur.lastrowid
        conn.commit()
        return cid
    finally:
        conn.close()


def get_collision(collision_id):
    conn=get_conn(); row=conn.execute("SELECT * FROM collision_records WHERE id=?",(int(collision_id),)).fetchone(); conn.close(); return dict(row) if row else None


def confirm_collision(collision_id,status,confirmer,confirmer_id):
    if status not in {"confirmed","false_positive"}: return False
    conn=get_conn()
    try:
        row=conn.execute("SELECT * FROM collision_records WHERE id=?",(int(collision_id),)).fetchone()
        if not row: return False
        cur=conn.execute("UPDATE collision_records SET status=?,confirmer=?,confirmer_id=?,confirmed_time=? WHERE id=? AND status='pending'",(status,confirmer,confirmer_id,now_iso(),collision_id))
        changed=cur.rowcount>0
        if changed and status=="false_positive":
            im=conn.execute("SELECT customer_id FROM images WHERE id=?",(row["matched_image_id"],)).fetchone()
            cid=im["customer_id"] if im else None
            conn.execute(
                """INSERT INTO false_positive_pairs(query_sha256,query_phash,matched_image_id,matched_customer_id,query_copy_feature,query_copy_dim,confirmer_id,created_time)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(query_sha256,matched_image_id) DO UPDATE SET
                     query_phash=excluded.query_phash,matched_customer_id=excluded.matched_customer_id,
                     query_copy_feature=excluded.query_copy_feature,query_copy_dim=excluded.query_copy_dim,
                     confirmer_id=excluded.confirmer_id,created_time=excluded.created_time""",
                (row["query_sha256"],row["query_phash"],row["matched_image_id"],cid,row["query_copy_feature"],row["query_copy_dim"],confirmer_id,now_iso()),
            )
        conn.commit(); return changed
    finally: conn.close()


def record_profile_conflict(customer_id,image_id,incoming_raw_text,conflicts,source="history"):
    if not conflicts: return
    conn=get_conn(); conn.execute("INSERT INTO customer_profile_conflicts(customer_id,image_id,incoming_raw_text,conflict_json,source,created_time) VALUES(?,?,?,?,?,?)",(customer_id,image_id,incoming_raw_text,json.dumps(conflicts,ensure_ascii=False),source,now_iso())); conn.commit(); conn.close()


def reindex_missing_signatures(limit=20):
    conn=get_conn()
    rows=conn.execute("""SELECT i.id,i.file_path FROM images i LEFT JOIN image_signatures s ON s.image_id=i.id WHERE s.image_id IS NULL ORDER BY i.id LIMIT ?""",(int(limit),)).fetchall()
    done=0
    try:
        for row in rows:
            if done>=int(limit):
                break
            p=_stored_path(row["file_path"])
            if not p: continue
            try:
                img=read_image(p); _save_signatures(conn,int(row["id"]),build_signatures(img)); done+=1
            except Exception: log.debug("reindex failed image=%s",row["id"],exc_info=True)
        conn.commit(); return done
    finally: conn.close()




def reindex_missing_copy_features(limit=6):
    """Background-upgrade old images with task-specific SSCD copy descriptors."""
    conn=get_conn()
    rows=conn.execute(
        """SELECT i.id,i.file_path FROM images i
           LEFT JOIN image_copy_features c ON c.image_id=i.id
           WHERE c.image_id IS NULL AND i.file_path IS NOT NULL
           ORDER BY i.id DESC LIMIT ?""",
        (max(int(limit)*20,100),),
    ).fetchall()
    done=0
    try:
        if not rows:
            return 0
        from ai.copy_embedding import extract_copy_features
        for row in rows:
            if done>=int(limit):
                break
            p=_stored_path(row["file_path"])
            if not p:
                continue
            try:
                img=read_image(p)
                views=extract_copy_features(img,max_views=4)
                if views:
                    save_copy_features(conn,int(row["id"]),views)
                    done+=1
            except Exception:
                log.debug("SSCD reindex failed image=%s",row["id"],exc_info=True)
        conn.commit()
        return done
    finally:
        conn.close()


def count_copy_index_status():
    conn=get_conn()
    try:
        total=int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] or 0)
        indexed=int(conn.execute("SELECT COUNT(DISTINCT image_id) FROM image_copy_features").fetchone()[0] or 0)
        return total,indexed,max(0,total-indexed)
    finally:
        conn.close()


def cleanup_orphan_image_files(max_age_hours=24):
    """Remove query/suspect temp images that were never added to the DB or pending buffer."""
    import time
    cutoff=time.time()-float(max_age_hours)*3600
    conn=get_conn()
    try:
        refs=set()
        for r in conn.execute("SELECT file_path FROM images WHERE file_path IS NOT NULL").fetchall():
            p=_stored_path(r["file_path"])
            if p: refs.add(str(p.resolve()))
        for r in conn.execute("SELECT file_path FROM pending_buffer WHERE file_path IS NOT NULL").fetchall():
            try: refs.add(str(Path(r["file_path"]).resolve()))
            except Exception: pass
    finally:
        conn.close()
    deleted=0
    root=Path(IMAGE_DIR)
    if not root.exists(): return 0
    for p in root.rglob("*"):
        if not p.is_file(): continue
        try:
            if str(p.resolve()) in refs: continue
            if p.stat().st_mtime >= cutoff: continue
            p.unlink(missing_ok=True); deleted+=1
        except Exception: pass
    return deleted


def save_customer_only(submitter, submitter_id, chat_id, customer_data, raw_text, source="history", source_message_id=""):
    conn=get_conn(); cur=conn.execute("""INSERT INTO customers(name,age,job,income,work_year,software,receiver,raw_text,submitter,submitter_id,chat_id,source,source_message_id,created_time) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(customer_data.get("name",""),customer_data.get("age",""),customer_data.get("job",""),customer_data.get("income",""),customer_data.get("work_year",""),customer_data.get("software",""),customer_data.get("receiver",""),raw_text or "",submitter,submitter_id,chat_id,source,str(source_message_id or ""),now_iso())); cid=cur.lastrowid; conn.commit(); conn.close(); return cid


def update_customer_fields(customer_id:int,fields:dict,submitter_id:str|None=None,operator:str|None=None,operator_id:str|None=None)->bool:
    allowed={"name","age","job","income","work_year","software","receiver"}; updates={k:str(v).strip() for k,v in fields.items() if k in allowed}
    if not updates:return False
    conn=get_conn()
    try:
        old=conn.execute("SELECT * FROM customers WHERE id=?",(customer_id,)).fetchone(); oldd=dict(old) if old else {}; ts=now_iso(); op=operator or submitter_id or "unknown"; opid=operator_id or submitter_id
        set_clause=", ".join(f"{k}=?" for k in list(updates)+["updated_time","last_updated_by"]); vals=list(updates.values())+[ts,op]
        if submitter_id is not None: sql=f"UPDATE customers SET {set_clause} WHERE id=? AND submitter_id=?"; vals += [customer_id,str(submitter_id)]
        else: sql=f"UPDATE customers SET {set_clause} WHERE id=?"; vals += [customer_id]
        cur=conn.execute(sql,vals)
        if cur.rowcount:
            for k,v in updates.items(): conn.execute("INSERT INTO customer_edit_log(customer_id,field_name,old_value,new_value,operator,operator_id,changed_time) VALUES(?,?,?,?,?,?,?)",(customer_id,k,str(oldd.get(k) or ""),v,op,opid,ts))
        conn.commit(); return cur.rowcount>0
    finally: conn.close()


def get_customer_by_id(customer_id:int):
    conn=get_conn(); row=conn.execute("SELECT * FROM customers WHERE id=?",(int(customer_id),)).fetchone(); conn.close(); return dict(row) if row else None


def get_image_by_id(image_id: int):
    """Return one stored image row plus a resolved local path for manual review."""
    conn=get_conn()
    try:
        row=conn.execute(
            """SELECT i.*, c.name AS customer_name
               FROM images i LEFT JOIN customers c ON c.id=i.customer_id
               WHERE i.id=? LIMIT 1""",
            (int(image_id),),
        ).fetchone()
        if not row:
            return None
        data=dict(row)
        rp=_stored_path(data.get("file_path"))
        data["resolved_path"]=str(rp) if rp else ""
        return data
    finally:
        conn.close()
