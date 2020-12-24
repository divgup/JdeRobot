import threading
import time
import math
import rosbag
import cv2
import numpy as np
from datetime import datetime


time_cycle = 40 #80

class MyAlgorithm(threading.Thread):

    def __init__(self, bag_readings, pose_obj):
        self.bag_readings = bag_readings
        self.pose_obj = pose_obj
        self.threshold_image = np.zeros((640,480,3), np.uint8)
        self.color_image = np.zeros((640,480,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
        self.diff_time = 0

        
    def getReadings(self , *sensors):
        self.lock.acquire()
        data = self.bag_readings.getData(sensors)
        self.lock.release()
        return data

    def set_predicted_path(self,path):
        self.pose_lock.acquire()
        
        self.pose_obj.set_pred_path(path)
        self.pose_lock.release()

    def set_predicted_pose(self,x,y,t):
        self.pose_lock.acquire()
        self.predicted_pose = [x,y]
        self.pose_obj.set_pred_pose([x,y],t)
        self.pose_lock.release()

    def get_predicted_pose(self):
        self.pose_lock.acquire()
        
    def set_processed_image(self,image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
    def get_processed_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        #self.algo_start_time = time.time()
        while (not self.kill_event.is_set()):
            start_time = datetime.now()

            if not self.stop_event.is_set():
                self.algo_start_time = time.time()
                self.algorithm()
                self.algo_stop_time = time.time()
                self.diff_time = self.diff_time + (self.algo_stop_time - self.algo_start_time)
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):


        fx = 525.0  # focal length x
        fy = 525.0  # focal length y
        cx = 319.5  # optical center x
        cy = 239.5  # optical center y

        # factor = 5000 # for the 16-bit PNG files
        factor = 1 # for the 32-bit float images in the ROS bag files


        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        lk_params = dict( winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # print(p0)
        # mask = np.zeros_like(old_frame)
        C_k_1 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        x = 0
        y = 0
        cnt=0
        tracks = []
        track_len = 10
        color = (255, 255, 0) 
        lst = []
        while(cnt < 100):
            
            # cnt+=1   
            # print(cnt) 
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data = self.getReadings('color_img' , 'depth_img','scan') # to get readings data from particular sensors
            depth_image = data.depth_img    
        #     ret,frame = cap.read()
            new_gray = cv2.cvtColor(data.color_img, cv2.COLOR_BGR2GRAY) 
            copy_new = data.color_img.copy()
            # copy_newga
        #     # calculate optical flow

            if(cnt > 0):
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)                
                # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)         
                    cv2.circle(copy_new, (x, y), 10, (120, 255, 0), -1)      
                    # cv2.circle()
        #     # Select good points
                tracks = new_tracks
                # cv2.polylines(copy_new,[np.int32(tr) for tr in tracks],False,color)
                # good_new = p1[st==1]
                # good_old = p0[st==1]
                # print("p0 ",np.array(p0).shape) 
                # print("p1 ",np.array(p1).shape)
                p3d_k_1 = []
                p3d_k = []
                # print(good)
                p_k_1 = np.squeeze(p0[good > 0])
                p_k = np.squeeze(p1[good > 0])
                # print("p_k_1 ",p_k_1.shape)
                for (pt1,pt2) in zip(p_k_1,p_k):              
                    Z1 = depth_image[int(pt1[1]),int(pt1[0])]/factor
                    Z2 = depth_image[int(pt2[1]),int(pt2[0])]/factor

                    if np.isnan(Z1) or np.isnan(Z2):
                        continue
                    X = (pt1[0]-cx)*Z1/fx
                    Y = (pt1[1]-cy)*Z1/fy
                    p3d_k_1.append([X,Y,Z1])
                    
                    X = (pt2[0]-cx)*Z2/fx 
                    Y = (pt2[1]-cy)*Z2/fy 
                    p3d_k.append([X,Y,Z2])   
                # print(np.array(p3d_k).shape)
                # print("P3d_k1 ",np.array(p3d_k_1).shape)    
                p3d_k= np.array(p3d_k)
                p3d_k_1 = np.array(p3d_k_1)
                # print(p3d_k[0:10])
                # print(p3d_k_1[0:10])
                num_3d_pts = p3d_k.shape[0]    
                K = 3
                # print("p3d",p3d_k[:K,])
                # print(np.mean(p3d_k[:K,],axis=0))
                A = p3d_k[:K,] - np.mean(p3d_k[:K,],axis=0)
                # print("matrix A ",A)
                B = p3d_k_1[:K,] - np.mean(p3d_k_1[:K,],axis=0)
                # print("matrix B ",B)
                svdmat = np.dot(B,A.T)
                # print("SVDMAT ",svdmat)
                [U,S,V] = np.linalg.svd(svdmat,full_matrices=0)
                # print("isPsd ",np.linalg.eigvals(svdmat))
                # print(S)
                # print("U ",U.shape)
                # print("V ",V.shape)
                # print("V_act ",V)
                R = np.dot(V.T,U.T)
                # print("R ",R)
                j=0
                inliers = 0
                while(inliers < 3 or np.linalg.det(R) < 0):                    
                    A = p3d_k[j:K+j,] - np.mean(p3d_k[j:K+j,],axis=0)
                    # print("matrix A ",A)
                    B = p3d_k_1[j:K+j,] - np.mean(p3d_k_1[j:K+j,],axis=0)
                    # print("matrix B ",B)
                    svdmat = np.dot(B,A.T)
                    # print("SVDMAT ",svdmat)
                    [U,S,V] = np.linalg.svd(svdmat,full_matrices=0)   
                    R = np.dot(V.T,U.T)
                    tvecs = np.mean(p3d_k[j:K+j,],axis=0)[0:3].reshape(3,1) - np.dot(R,np.mean(p3d_k_1[j:K+j,0:3],axis=0).T).reshape(3,1)
                    pts_transformed = np.dot(R,p3d_k_1.T) + tvecs
                    pts_transformed = pts_transformed.T
                    inliers=0
                    for leng in range(3):
                        # print(np.linalg.norm(pts_transformed[leng] - p3d_k[leng]))
                        if(np.linalg.norm(pts_transformed[leng] - p3d_k[leng])) < 0.15:
                            inliers+=1
                    j+=1
                    # print("R_checking ",R)
                    # print("detR ",np.linalg.det(R))

                # if(np.linalg.det(R) < 0):
                #     # print("V_lastcol ",-V.T[:,K-1])
                #     last_col =  np.reshape(-V.T[:,K-1],(3,1))
                #     V_dash = np.hstack((V.T[:,0:(K-1)],last_col))
                #     # print("V_dash",V_dash)
                #     R = np.dot(V_dash.T,U.T)
                # print("Rot ",R)
                # print("Det",np.linalg.det(R))
                # print("shape ",np.mean(p3d_k_1[j:K+j,0:3],axis=0).T.shape)
                tvecs = np.mean(p3d_k[j:K+j,],axis=0)[0:3].reshape(3,1) - np.dot(R,np.mean(p3d_k_1[j:K+j,0:3],axis=0).T).reshape(3,1)
                # print("Translation ",tvecs)
                mat = np.dot(R,svdmat)
                # print("isPsd ",np.linalg.eigvals(mat))
                # A = np.zeros((K*3,12))    
                # for i in range(K*3):
                #     if(i%3==0):
                #         ptr=0
                #     else:
                #         ptr+=4
                #     for j in range(0,4):
                #         A[i,ptr + j] = p3d_k_1[i/3,j]
                # print(A)
                # [U,S,V] = np.linalg.svd(A,full_matrices=0)
                # S = np.array(S)
                # print("Sigma_inv",np.linalg.inv(np.diag(S)))
                # print("V",V.shape)
                # print("S",S.shape)
                # print("U",U.shape)
                # invA = np.dot(V.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
                # poseinfo = np.dot(invA,np.reshape(p3d_k[:K],(-1,1)))
                # for i in range(3):
                Pose = np.hstack((R,tvecs))
                Pose = np.reshape(Pose,(3,4))
                Pose = np.vstack((Pose,[0,0,0,1]))
                # print(Pose)                
                C_k = np.dot(C_k_1,Pose)
                tvecs = C_k[:,3]
                print("C_k",C_k)
                x = tvecs[0]
                y = tvecs[1]     
                # print("(x,y) ",x,y)        
                C_k_1 = C_k    
            # self.set_predicted_pose(x,y,color_img_t)    
            lst.append([x,y])
            arr =  np.reshape(np.array(lst),(-1,2))     
            color_img_t = data.color_img_t 
            # self.set_predicted_pose(x,y,color_img_t) 
            if(cnt%10 == 0):
                mask = np.zeros_like(new_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p0 = cv2.goodFeaturesToTrack(new_gray, mask = mask, **feature_params)
                if p0 is not None:
                    for x, y in np.float32(p0).reshape(-1, 2):
                        tracks.append([(x, y)])                
        #     # Now update the previous frame and previous points
            old_gray = new_gray.copy()
            color_image = copy_new
            #depth image
            depth_image = data.depth_img
            # print(depth_image.dtype)
            
            scan_d = data.scan
            # cv2.imwrite("/home/divanshu05/Academy/exercises/2d_visual_odometry/depth_img{}.jpg".format(cnt),depth_image)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # print(((5 < depth_image) & (depth_image < 10)).sum())
            # print("Scan",np.array(scan_d).shape)
            # print()
            # for v in range(depth_image.height):
            #     for u in range(depth_image.width):
            #         Z = depth_image[v,u] / factor
            #         X = (u - cx) * Z / fx
            #         Y = (v - cy) * Z / fy
            

            #Show processed image on GUI
            self.set_processed_image(color_image)
            #self.set_processed_image(depth_image)
            #set predicted pose

            


            # print("arr",arr)
            # self.set_predicted_path(arr)
            cnt+=1

        #set predicted path at once /or reset the previously set predicted poses at once ---- path should be Nx2 numpy array or python list [x,y].
        
