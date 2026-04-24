import cv2
import numpy as np
#-------------------------------------------------
#任务1  ORB 特征检测与描述子计算
# 读取图像
box_img = cv2.imread('box.png')
box_in_scene_img = cv2.imread('box_in_scene.png')

# 检查图像是否加载成功
if box_img is None:
    print("Error: Could not load box.png")
    exit(1)
if box_in_scene_img is None:
    print("Error: Could not load box_in_scene.png")
    exit(1)

# 创建 ORB 检测器，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 检测 box.png 的关键点和描述子
keypoints_box, descriptors_box = orb.detectAndCompute(box_img, None)

# 检测 box_in_scene.png 的关键点和描述子
keypoints_scene, descriptors_scene = orb.detectAndCompute(box_in_scene_img, None)

# 检查是否检测到关键点和描述子
if keypoints_box is None or descriptors_box is None:
    print("Error: No keypoints or descriptors found in box.png")
    exit(1)
if keypoints_scene is None or descriptors_scene is None:
    print("Error: No keypoints or descriptors found in box_in_scene.png")
    exit(1)

# 可视化关键点
box_with_keypoints = cv2.drawKeypoints(box_img, keypoints_box, None)
scene_with_keypoints = cv2.drawKeypoints(box_in_scene_img, keypoints_scene, None)

# 保存可视化图像
cv2.imwrite('box_keypoints.png', box_with_keypoints)
cv2.imwrite('box_in_scene_keypoints.png', scene_with_keypoints)

# 输出关键点数量
print(f"box.png 中的关键点数量: {len(keypoints_box)}")
print(f"box_in_scene.png 中的关键点数量: {len(keypoints_scene)}")

# 输出描述子维度
print(f"描述子维度: {descriptors_box.shape[1]}")  # ORB 描述子是 32 维


#------------------------------------------------
#任务2  ORB 特征匹配并可视化匹配结果

# 创建 BFMatcher，使用 NORM_HAMMING 和 crossCheck=True
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 进行匹配
matches = bf.match(descriptors_box, descriptors_scene)

# 按照匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 输出总匹配数量
print(f"总匹配数量: {len(matches)}")

# 可视化前 50 个匹配结果
img_matches = cv2.drawMatches(box_img, keypoints_box, box_in_scene_img, keypoints_scene, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 保存 ORB 匹配可视化图像
cv2.imwrite('orb_matches_top50.png', img_matches)

#-----------------------------------------------
#任务3  RANSAC 剔除错误匹配

# 从匹配结果中提取对应点坐标
src_pts = np.float32([keypoints_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 使用 cv2.findHomography() 估计单应矩阵，使用 RANSAC，设置重投影误差阈值 5.0
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 根据 mask 筛选内点匹配
inliers = [m for i, m in enumerate(matches) if mask[i]]

# 输出 Homography 矩阵
print("Homography 矩阵:")
print(M)

# 输出总匹配数量
print(f"总匹配数量: {len(matches)}")

# 输出 RANSAC 内点数量
num_inliers = np.sum(mask)
print(f"RANSAC 内点数量: {num_inliers}")

# 输出内点比例
inlier_ratio = num_inliers / len(matches)
print(f"内点比例: {inlier_ratio:.4f}")

# 可视化 RANSAC 后的内点匹配（显示前50个）
img_ransac = cv2.drawMatches(box_img, keypoints_box, box_in_scene_img, keypoints_scene, inliers[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 保存 RANSAC 匹配可视化图像
cv2.imwrite('ransac_matches.png', img_ransac)

#------------------------------------------------
#任务4  目标定位

# 获取 box.png 的四个角点
h, w = box_img.shape[:2]
box_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)

# 使用 cv2.perspectiveTransform() 进行角点投影
dst_corners = cv2.perspectiveTransform(box_corners, M)

# 在场景图中画出四边形边框
scene_copy = box_in_scene_img.copy()
cv2.polylines(scene_copy, [dst_corners.astype(int)], True, (0, 255, 0), 3, cv2.LINE_AA)

# 保存目标定位结果图
cv2.imwrite('target_localization.png', scene_copy)

# 显示最终目标定位结果
print("目标定位成功：在 box_in_scene.png 中画出了目标物体的边框")

#------------------------------------------------
#任务6  参数对比实验

print("\nnfeatures\t模板图关键点数\t场景图关键点数\t匹配数量\tRANSAC内点数\t内点比例\t是否成功定位")

nfeatures_list = [500, 1000, 2000]

for nfeat in nfeatures_list:
    # 创建 ORB 检测器
    orb = cv2.ORB_create(nfeatures=nfeat)
    
    # 检测关键点和描述子
    keypoints_box, descriptors_box = orb.detectAndCompute(box_img, None)
    keypoints_scene, descriptors_scene = orb.detectAndCompute(box_in_scene_img, None)
    
    # 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_box, descriptors_scene)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # RANSAC
    if len(matches) >= 4:  # 需要至少4个点来估计Homography
        src_pts = np.float32([keypoints_box[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        num_inliers = np.sum(mask)
        inlier_ratio = num_inliers / len(matches)
        success = M is not None and num_inliers > 0
    else:
        M = None
        mask = None
        num_inliers = 0
        inlier_ratio = 0.0
        success = False
    
    # 打印结果
    print(f"{nfeat}\t\t{len(keypoints_box)}\t\t{len(keypoints_scene)}\t\t{len(matches)}\t\t{num_inliers}\t\t{inlier_ratio:.4f}\t\t{success}")

# 比较分析
print("\n比较分析：")
print("1. 不同 nfeatures 对匹配数量的影响：随着 nfeatures 增加，关键点数量增加，导致匹配数量总体增加，但匹配质量可能下降。")
print("2. 不同 nfeatures 对 RANSAC 内点比例的影响：内点比例不一定随关键点增加而提高，可能因噪声增加而降低。")
print("3. 特征点越多，定位效果不一定越好：适当数量的关键点能提供足够信息，过多可能引入更多错误匹配，降低定位精度。")

#------------------------------------------------
#选做任务：SIFT 特征匹配

try:
    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述子
    keypoints_box_sift, descriptors_box_sift = sift.detectAndCompute(box_img, None)
    keypoints_scene_sift, descriptors_scene_sift = sift.detectAndCompute(box_in_scene_img, None)
    
    # 使用 BFMatcher 和 NORM_L2
    bf_sift = cv2.BFMatcher(cv2.NORM_L2)
    
    # KNN matching, k=2
    knn_matches = bf_sift.knnMatch(descriptors_box_sift, descriptors_scene_sift, k=2)
    
    # Lowe ratio test
    good_matches_sift = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches_sift.append(m)
    
    # RANSAC + Homography
    if len(good_matches_sift) >= 4:
        src_pts_sift = np.float32([keypoints_box_sift[m.queryIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
        dst_pts_sift = np.float32([keypoints_scene_sift[m.trainIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
        M_sift, mask_sift = cv2.findHomography(src_pts_sift, dst_pts_sift, cv2.RANSAC, 5.0)
        num_inliers_sift = np.sum(mask_sift)
        inlier_ratio_sift = num_inliers_sift / len(good_matches_sift)
        success_sift = M_sift is not None and num_inliers_sift > 0
    else:
        num_inliers_sift = 0
        inlier_ratio_sift = 0.0
        success_sift = False
    
    # 输出 SIFT 结果
    print(f"\nSIFT 匹配数量: {len(good_matches_sift)}")
    print(f"SIFT RANSAC 内点数: {num_inliers_sift}")
    print(f"SIFT 内点比例: {inlier_ratio_sift:.4f}")
    print(f"SIFT 是否成功定位: {success_sift}")
    
    # 对比表格
    print("\n对比表格：")
    print("方法\t匹配数量\tRANSAC内点数\t内点比例\t是否成功定位\t运行速度主观评价")
    print(f"ORB\t{len(matches)}\t{num_inliers}\t{inlier_ratio:.4f}\t{success}\t快")
    print(f"SIFT\t{len(good_matches_sift)}\t{num_inliers_sift}\t{inlier_ratio_sift:.4f}\t{success_sift}\t慢")
    
except Exception as e:
    print(f"\nSIFT 不支持或出错: {e}")
    print("跳过 SIFT 实验")

