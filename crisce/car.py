import copy
import itertools
import seaborn as sns
import glob
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import numpy as np
from pre_processing import Pre_Processing
import cv2
import scipy.special
import time
import matplotlib


vehicle_info = "vehicle_info"

class Car():

    def __init__(self):
        self.car_length, self.car_width, self.center_of_vehicles = list(), list(), list()
        self.vehicles = dict()
        self.vehicles["red"]    = dict()
        self.vehicles["blue"]   = dict()
        self.pre_process    = Pre_Processing()
        self.car_length     = None
        self.car_width      = None
        self.car_length_sim = None
        self.height         = None
        self.width          = None
        self.show_image     = None
        self.output_folder  = None
        self.process_number = None
        
    def getCarDimensions(self):
        """ Return car length and car width"""
        return (self.car_length, self.car_width)
    
    def getImageDimensions(self):
        """ Return heigth and width"""
        return (self.height, self.width)

    def edgesofTheCar(self, box_points, width_of_car):
        # rect_points = list()
        points_on_vehicle = box_points.tolist()
        advance_points = list()
        points_on_width = list()
        x = list(range(4))
        y = list(range(1,4))
        y.append(0)
        nodes_coverage = list(zip(x,y))
        for i,j in nodes_coverage:
            # print(box_points, box_points[i][0])
            ed_edges = np.sqrt(np.square(np.array(box_points[j][0]) - np.array(box_points[i][0])) + np.square(np.array(box_points[j][1]) - np.array(box_points[i][1])))
            # print("\n width_of_car ", width_of_car)
            # print("ed edges of the car ", ed_edges)

            if(ed_edges > width_of_car * 1.1):
                """ We need to find three points on the side of a vehicle """
                mid_point = [(int(box_points[i][0]) + int(box_points[j][0])) // 2, (int(box_points[i][1]) + int(box_points[j][1])) // 2]

                mid_point_1 = [(int(mid_point[0]) + int(box_points[i][0])) // 2, (int(mid_point[1]) + int(box_points[i][1])) // 2]

                mid_point_2 = [(int(mid_point[0]) + int(box_points[j][0])) // 2, (int(mid_point[1]) + int(box_points[j][1])) // 2]
                advance_points.extend([mid_point_1, mid_point_2])
            else:
                mid_point = [(int(box_points[i][0]) + int(box_points[j][0])) // 2, (int(box_points[i][1]) + int(box_points[j][1])) // 2]
                points_on_width.append(mid_point)
                # rect_points.append(mid_point)
            points_on_vehicle.append(mid_point)
        # last_midpoint_point = [(int(box_points[0][0]) + int(box_points[-1][0])) // 2, (int(box_points[0][1]) + int(box_points[-1][1])) // 2]
        # points_on_vehicle.append(last_midpoint_point)
        points_on_vehicle.extend(advance_points)
        return points_on_vehicle, points_on_width


    def extractVehicleContoursFromMask(self):
        for vehicle_color in self.vehicles:
            contours, hierarchy = cv2.findContours(self.vehicles[vehicle_color]["mask"].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.vehicles[vehicle_color]["contour"] = contours
            # print(len(self.vehicles[vehicle_color]["contour"]), hierarchy)
            self.vehicles[vehicle_color].pop('mask', None)

    def geometricOperationOnVehicle(self, image, time_efficiency):
        """ Finds number of things related to the vehicles:
        
        1. min_area_rect (OBB == Object Oriented Bounding Box)(returns the center, (width, height), angle)
        2. center_of_car (center of vehicle in image)
        3. angle_of_rect (orientation of rectangle in image)
        4. vehicle_nodes (8 points on vehicle including box_points)
        5. car_length
        6. car_width

        Args:
            vehicles ([dictionary]): [information about vehicles]
        """
        c_length, c_width, center_of_vehicles = list(), list(), list()
        vehicle_info = "vehicle_info"
        
        t0 = time.time()
        print("\n-------  Vehicle Extraction Pipeline    ------")
        print("\n-------  Extracting Geometric Information of vehicles   ------")
        for vehicle_color in self.vehicles:
            print("\n{} Vehicle \n".format(str.capitalize((vehicle_color))))
            self.vehicles[vehicle_color][vehicle_info] = dict()
            count_id = 0
            for i, contour in enumerate(self.vehicles[vehicle_color]["contour"]):
                # print ("Vehicle  {} # {},  Shape = {},  Area = {},  Arc_Length = {} ".format(vehicle_color, i, contour.shape, 
                #                                                                              cv2.contourArea(contour), cv2.arcLength(contour, closed=False)))
                print ("Vehicle  {} # {},  Area = {},  Arc_Length = {} ".format(vehicle_color, i, 
                                                                                cv2.contourArea(contour), cv2.arcLength(contour, closed=False)))

                
                if cv2.contourArea(contour) > 60:
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)] = dict()
                    self.vehicles[vehicle_color]["dimensions"] = dict()

                    M = cv2.moments(contour)
                    # self.vehicles[vehicle_color]["moments"] = M
                    # print("Moments of contour are = ", M)

                    # mid_x = M["m10"] / M["m00"]
                    # mid_y = M["m01"] / M["m00"]
                    # self.vehicles["blue"]["center_of_car"] = (mid_x, mid_y)
                    # print("vehicles center_of_car = ", self.vehicles["blue"]["center_of_car"])

                    x, y, width, height = cv2.boundingRect(contour)
                    rect = cv2.minAreaRect(contour)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    img = cv2.polylines(image.copy(), [box], True, (0,255,0), 2)
                    
                    temp_img = cv2.rectangle(image.copy(), (x,y), (x + width, y + height), (0,204,255), 2)
                    # temp_img = cv2.line(image.copy(), tuple([box[1][0], box[1][1]]), tuple([box[2][0], box[2][1]]), (128,128,128), 3)
                    # cv2.circle(img, tuple([int(rect[0][0]), int(rect[0][1])]), 5, (204, 0, 204), -1)

                    ### cv2.norm(box[0] - box[1], cv2.NORM_L2)
                    ## square_root((x2-x1)**2 + (y2-y1)**2)
                    dist_p_12 = math.sqrt(math.pow(int(round(box[0][0])) - int(round(box[1][0])), 2) + math.pow(int(round(box[0][1])) - int(round(box[1][1])), 2))
                    ## square_root((x3-x2)**2 + (y3-y2)**2)
                    dist_p_23 = math.sqrt(math.pow(int(round(box[1][0])) - int(round(box[2][0])), 2) + math.pow(int(round(box[1][1])) - int(round(box[2][1])), 2))
                    
                    c_length.append(max(dist_p_12, dist_p_23))
                    c_width.append(min(dist_p_12, dist_p_23))
                    
                    center_of_vehicles.append((int(rect[0][0]), int(rect[0][1])))

                    vehicle_nodes, nodes_on_width = self.edgesofTheCar(box, min(dist_p_12, dist_p_23))
                    vehicle_center = ((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / len(box), 
                                    (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / len(box))
                    # cv2.circle(img, tuple([int(vehicle_center[0]), int(vehicle_center[1])]), 8, (0, 255, 255), -1)

                    # print(edgesofTheCar(box))
                    # print("vehicle_nodes array", np.array(vehicle_nodes))

                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["bounding_rect"]  = cv2.boundingRect(contour)
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["min_area_rect"]  = rect
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["center_of_car"]  = (rect[0][0], rect[0][1]) # vehicle_center  # /int(rect[0][0], int(rect[0][1]))
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["angle_of_rect"]  = rect[2]
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["vehicle_nodes"]  = vehicle_nodes
                    self.vehicles[vehicle_color][vehicle_info][str(count_id)]["nodes_on_width"] = nodes_on_width

                    """ Corner Points and Nodes of Vehicles """
                    # cv2.rectangle(img, (x, y),(x + width, y + height), (255, 255, 0), -1)
                    cv2.circle(img, tuple([int(vehicle_nodes[0][0]), int(vehicle_nodes[0][1])]), 5, (255, 255, 0), -1)  ## Light Sky blue   , Node 0
                    cv2.circle(img, tuple([int(vehicle_nodes[1][0]), int(vehicle_nodes[1][1])]), 5, (127, 0, 255), -1)  ## Lipstic Pink     , Node 1
                    cv2.circle(img, tuple([int(vehicle_nodes[2][0]), int(vehicle_nodes[2][1])]), 5, (255, 51, 153), -1) ## Purple color     , Node 2
                    cv2.circle(img, tuple([int(vehicle_nodes[3][0]), int(vehicle_nodes[3][1])]), 5, (255, 0, 0), -1)    ## Dark Blue        , Node 3
                    cv2.circle(img, tuple([int(vehicle_nodes[4][0]), int(vehicle_nodes[4][1])]), 5, (50, 255, 255), -1) ## Yellow           , Node 4 (Mid_0_1)
                    cv2.circle(img, tuple([int(vehicle_nodes[5][0]), int(vehicle_nodes[5][1])]), 5, (50, 255, 255), -1) ## Yellow           , Node 5 (Mid_1_2)
                    cv2.circle(img, tuple([int(vehicle_nodes[6][0]), int(vehicle_nodes[6][1])]), 5, (50, 255, 255), -1) ## Yellow           , Node 6 (Mid_2_3)
                    cv2.circle(img, tuple([int(vehicle_nodes[7][0]), int(vehicle_nodes[7][1])]), 5, (50, 255, 255), -1) ## Yellow           , Node 7 (Mid_3_4)
                    cv2.circle(img, tuple([int(vehicle_nodes[8][0]), int(vehicle_nodes[8][1])]), 5, (0, 255, 0), -1)    ## Green            , Node 8 (Mid_4_1)
                    cv2.circle(img, tuple([int(vehicle_nodes[9][0]), int(vehicle_nodes[9][1])]), 5, (0, 255, 0), -1)    ## Green            , Node 9 (Mid_4_2)
                    cv2.circle(img, tuple([int(vehicle_nodes[10][0]), int(vehicle_nodes[10][1])]), 5, (0, 255, 0), -1)  ## Green            , Node 10 (Mid_7_3)
                    cv2.circle(img, tuple([int(vehicle_nodes[11][0]), int(vehicle_nodes[11][1])]), 5, (0, 255, 0), -1)  ## Green            , Node 11 (Mid_7_4)

                    count_id += 1     
                    
                    # print("vehicles center_of_car = {} \n".format(self.vehicles[vehicle_color][vehicle_info][str(count_id)]["center_of_car"]))
                    
                    # if self.show_image:
                        # self.pre_process.showImage(" Axis Aligned Bounding Boxes Vs Oriented Bounding Boxes", np.hstack([temp_img, img]))
            self.vehicles[vehicle_color].pop("contour", None)
        
        # c_length.sort()
        # c_width.sort()
        # self.car_length = round(sum(c_length))
        # self.car_width  = round(c_width)
        
        self.car_length = sum(c_length) / len(c_length)
        self.car_width  = sum(c_width) / len(c_length)
        
        # self.car_length = round(min(c_length)) # round(max(c_length)),, default -> round(max(c_length))
        # self.car_width  = round(max(c_width))
        
        for vehicle_color in self.vehicles:
            self.vehicles[vehicle_color]["dimensions"]["car_length"]     = self.car_length
            self.vehicles[vehicle_color]["dimensions"]["car_length_sim"] = self.car_length_sim
            self.vehicles[vehicle_color]["dimensions"]["car_width"]      = self.car_width
            
        t1 = time.time()
        
        print("\n---- Dimensions of Vehicles -------\n")
        print("car_length = ", self.car_length)
        print("car_width  = ", self.car_width)
        
        if self.show_image:
            self.pre_process.showImage("Axis Aligned Bounding Boxes Vs Oriented Bounding Boxes", np.hstack([temp_img, img]), time=800)

        cv2.imwrite(self.output_folder + "{}_AABB_OBB.jpg".format(self.process_number), np.hstack([temp_img, img]))
        self.process_number += 1
        time_efficiency["calc_vehicle_nodes"] = t1-t0
        # print("total time taken", t1-t0)


    def extractingCrashPoint(self, image, time_efficiency):
        """ Distance calculation between vehicles nodes and extracting the crash point """
        
        t0 = time.time()
        nodes_of_vehicles = [self.vehicles[vehicle_color]["vehicle_info"][str(v_id)]["vehicle_nodes"] 
                            for vehicle_color in self.vehicles for v_id in self.vehicles[vehicle_color]["vehicle_info"]]
        
        crash_ed_cal = list()

        for vehicle1_nodes, vehicle2_nodes in itertools.combinations(nodes_of_vehicles, 2):

            point_dist = list()
            crash_img = image.copy()
            for point_1 in vehicle1_nodes:
                cv2.circle(crash_img, tuple(point_1), radius=3,
                        color=(0, 255, 0), thickness=-1)
                for point_2 in vehicle2_nodes:
                    cv2.circle(crash_img, tuple(point_2), radius=3,
                            color=(255, 0, 0), thickness=-1)
                    crash_ed = cv2.norm(np.array(point_1) -
                                        np.array(point_2), cv2.NORM_L2)
                    # print("crash_ed", crash_ed)
                    point_dist.append([crash_ed, point_1, point_2])
                    # print("point_dist", point_dist)
                #     cv2.imshow("crash point analyzation", crash_img)
                # cv2.waitKey(50)
            # cv2.destroyAllWindows()
            # cv2.imshow("crash point analyzation", crash_img)
            # cv2.waitKey(0)
            crash_ed_cal.append(min(point_dist))

        # print("crash euclidean distance calculation for the two vehicles", crash_ed_cal)

        min_dist = min(crash_ed_cal)
        crash_point = [(int(round(min_dist[1][0])) + int(round(min_dist[2][0]))) // 2,
                       (int(round(min_dist[1][1])) + int(round(min_dist[2][1]))) // 2]

        for vehicle_color in self.vehicles:
            self.vehicles[vehicle_color]["crash_point"] = dict()
            self.vehicles[vehicle_color]["crash_point"]["coordinates"]      = crash_point
            self.vehicles[vehicle_color]["crash_point"]["dist_to_vehicle"]  = min_dist[0]
        
        t1 = time.time()
        
        cv2.circle(crash_img, tuple(crash_point), 8, (0, 128, 255), -1)
        # print("crash point = ", tuple(crash_point))
        cv2.imwrite(self.output_folder + "{}_crash_point_visualization.jpg".format(self.process_number), crash_img)
        self.process_number += 1

        if self.show_image:
            self.pre_process.showImage("crash point visualization", crash_img, time=800)
        
        time_efficiency["calc_crash_pt"] = t1-t0
        # print("total time taken", t1-t0)

        return crash_point

    def extractTriangle(self, image, time_efficiency):
        """
        Triangle Extraction
        """
        t0 = time.time()
        for vehicle_color in self.vehicles:
            for vehicle_id in self.vehicles[vehicle_color]['vehicle_info']:
                test = np.zeros_like(image) 
                test_image = image #.copy()
                x, y, w, h = self.vehicles[vehicle_color]['vehicle_info'][vehicle_id]["bounding_rect"]
                box_points = self.vehicles[vehicle_color]['vehicle_info'][vehicle_id]["vehicle_nodes"][:4]
                # ROI = image[y:y+h, x:x+w]
                roi = cv2.fillPoly(test, np.array([box_points]), (255, 255, 255))
                roi = cv2.bitwise_and(test_image, test_image, mask=roi[:,:,0])
                # cv2.imshow("ROI", roi)
                roi = cv2.bitwise_not(roi)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.blur(gray, (3, 3), 0)
                # cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
                _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
                thresh = cv2.bitwise_not(thresh)
                # if self.show_image:
                    # cv2.imshow("gray", gray)
                    # cv2.imshow("blur", blur)
                    # cv2.imshow("thresh", thresh)
                    # cv2.imshow("after applying NOT ", thresh)
                    # cv2.waitKey(400)

                # # kernel = np.zeros((4,4), np.uint8)        
                # # dilate = cv2.dilate(thresh, kernel)
                # # dilate = cv2.bitwise_not(dilate)
                # # rect_kernel = cv2.getStructuringElement( cv2.MORPH_RECT,(5,5))
                # # dilate = cv2.dilate(thresh, rect_kernel )
                cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt = max(cnts, key=cv2.contourArea)
                # print("number of selected contours", len(cnts))
                # print("")
                # print("contour shape ", cnt.shape)
                # # peri = cv2.arcLength(contour, True)
                # # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                # # # if the shape is a triangle, it will have 3 vertices
                # # print("len(approx)", len(approx))
                # # if len(approx) == 3:
                # #     print("triangle")
                rect = cv2.minAreaRect(cnt)
                # print("mid values", tuple([int(rect[0][0]), int(rect[0][1])]))
                # cv2.circle(img, tuple(self.vehicles[vehicle_color]["crash_point"]["coordinates"]), 5, (0, 128, 255), -1)
                cv2.circle(test_image, (int(rect[0][0]), int(rect[0][1])), 2, (255, 128, 0), -1 )
                self.vehicles[vehicle_color]['vehicle_info'][vehicle_id]["triangle_position"] = tuple([int(rect[0][0]), int(rect[0][1])])
                
        t1 = time.time()
        cv2.destroyAllWindows()
        cv2.imwrite(self.output_folder + "{}_triangle_extraction.jpg".format(self.process_number), test_image)
        self.process_number += 1
        
        time_efficiency["tri_ext"] = t1-t0
        # print("total time taken", t1-t0)
        

    def pointNearNodes(self, nodes, point):
        distances = np.sqrt(np.square(nodes[:, 0] - point[0]) + np.square(nodes[:, 1] - point[1]))
        distances = distances.reshape(distances.shape[0], -1)
        dist_nodes = np.hstack((distances, nodes))
        near_nodes = dist_nodes[dist_nodes[:, 0].argsort()]
        return near_nodes

    def extractingAnglesForVehicles(self, image, time_efficiency):
        t0 = time.time()
        print("\n-------- Extracting Angle of Vehicles -------\n")
        for vehicle_color in self.vehicles:
            print("{} Vehicle".format(str.capitalize(vehicle_color)))
            for vehicle_id in self.vehicles[vehicle_color]["vehicle_info"]:
                points_along_width = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["nodes_on_width"], np.int32)
                box = self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][:4]
                triangle_position = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["triangle_position"], np.int32)
                

                nodes_near_triangle = self.pointNearNodes(points_along_width, triangle_position)
                nodes_near_triangle = nodes_near_triangle[:,1:]

                point_1 = nodes_near_triangle[1]
                point_2 = nodes_near_triangle[0]

                x = [round(point_1[0]), round(point_2[0])]
                y = [round(point_1[1]), round(point_2[1])]
                dy = y[1] - y[0]
                dx = x[1] - x[0]
                rads = math.atan2(dy,dx)
                angle_of_vehicle = math.degrees(rads)
                
                ## Extrapolation based on the angle extracted
                length = 25 
                extrap_point_x =  int(round(point_2[0] + length * 1.1 * math.cos(angle_of_vehicle * np.pi / 180.0)))
                extrap_point_y =  int(round(point_2[1] + length * 1.1 * math.sin(angle_of_vehicle * np.pi / 180.0)))

                cv2.line(image, tuple( [int(point_2[0]), int(point_2[1])]), tuple([extrap_point_x, extrap_point_y]), (255, 255, 0), 2)  # (0, 100, 255)

                ### Orientation vehicle based on angle computed using position of the triangle
                angle_of_vehicle = (-angle_of_vehicle if angle_of_vehicle < 0 else ( -angle_of_vehicle + 360 ))
                print("V{}, Angle = {} ".format(vehicle_id, angle_of_vehicle))

                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["angle_of_car"] = angle_of_vehicle
                # cv2.polylines(image, np.array([box]), True, (36, 255, 12), 3)
                # cv2.imshow("Angles For Vehicles", image)
                # cv2.waitKey(0)
        
        t1 = time.time()

        if self.show_image:
            self.pre_process.showImage("Angles For Vehicles", image, time=800)

        cv2.imwrite(self.output_folder + "{}_angles_for_vehicles.jpg".format(self.process_number), image)
        self.process_number += 1
        
        time_efficiency["angle_cal"] = t1-t0
        # print("total time taken", t1-t0)
        

    def settingVehiclesInfo(self, v_color, vehicle_dist_pivot):
        del self.vehicles[v_color]["vehicle_info"]
        self.vehicles[v_color]["vehicle_info"] = dict()
        print("\n Arranging the snapshots for {} vehicle in order".format(v_color))
        for v_id in range(len(vehicle_dist_pivot)):
            # print("vehicle number =", v_id)
            self.vehicles[v_color]["vehicle_info"][str(v_id)] = vehicle_dist_pivot[int(v_id)][2]

    def extractVehicleProjectedSide(self, v_color, side):
        orient_count = 0
        side_name   = side[0]
        side_value = np.array(side[1])
        oriented_vehicles = list()
        
        for v_id in self.vehicles[v_color]["vehicle_info"]:
            # print(v_id, "angle of the car ", round(vehicles[v_color]["vehicle_info"][v_id]["angle_of_car"]))
            veh_angle = self.vehicles[v_color]["vehicle_info"][v_id]["angle_of_car"]
            points_along_width = np.asarray(self.vehicles[v_color]["vehicle_info"][v_id]["nodes_on_width"], np.int32)
            triangle_position  = np.asarray(self.vehicles[v_color]["vehicle_info"][v_id]["triangle_position"], np.int32)
            
            nodes_near_triangle = self.pointNearNodes(points_along_width, triangle_position)
            nodes_near_triangle = nodes_near_triangle[:, 1:]
            front_point = nodes_near_triangle[0]
            dx = side_value[:, 0] - front_point[0]
            dy = side_value[:, 1] - front_point[1]
            projected_angles    = np.arctan2(dy, dx)
            projected_angles    = projected_angles * 180 / np.pi
            side_results = [-round(angle) if angle < 0 else (-round(angle) + 360) for angle in projected_angles ]
            # print(side_results)
            # print("vehicle's angle exist on the projected side", side_name, "= ", round(veh_angle) in side_results)
            if round(veh_angle) in side_results:
                print("V{} is oriented towards the projected side = {}".format(v_id, side_name))
                oriented_vehicles.append(v_id)
                orient_count += 1
        return orient_count, oriented_vehicles

    def extractLeastDistanceVehicle(self, v_color):
        side_dist_calc = list()
        ### Least distant vehicle from the frame of the image
        for v_id in self.vehicles[v_color]["vehicle_info"]:
            waypoint_of_vehicle = self.vehicles[v_color]["vehicle_info"][v_id]["center_of_car"]
            Dy_min =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([waypoint_of_vehicle[0], 0]), cv2.NORM_L2) ## Top 
            Dx_min =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([0, waypoint_of_vehicle[1]]), cv2.NORM_L2) ## Left
            Dy_max =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([waypoint_of_vehicle[0], self.height]), cv2.NORM_L2) ## Bottom
            Dx_max =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([self.width, waypoint_of_vehicle[1]]), cv2.NORM_L2) ## Right
            self.vehicles[v_color]["vehicle_info"][v_id]["distance_from_boundary"] = min([(Dy_min, "top"), (Dx_min, "left"), (Dy_max, "bottom"), (Dx_max, "right")])
            side_dist_calc.append(self.vehicles[v_color]["vehicle_info"][v_id]["distance_from_boundary"])
            print("V{},  min distance from sketch boundary  = {}".format(v_id , self.vehicles[v_color]["vehicle_info"][v_id]["distance_from_boundary"]))
        print("V{} has the least distance from sketch boundary".format(side_dist_calc.index(min(side_dist_calc))))
        least_distant = str(side_dist_calc.index(min(side_dist_calc)))
        self.vehicles[v_color]["least_distant_vehicle"] = self.vehicles[v_color]["vehicle_info"][least_distant]
        
        return least_distant, side_dist_calc


    def sideDistances(self, v_color, v_id, projected_side):
        waypoint_of_vehicle = self.vehicles[v_color]["vehicle_info"][v_id]["center_of_car"]
        if projected_side == "TOP":
            dist =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([waypoint_of_vehicle[0], 0]), cv2.NORM_L2) ## Top 
        elif projected_side == "LEFT":
            dist =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([0, waypoint_of_vehicle[1]]), cv2.NORM_L2) ## Left
        elif projected_side == "BOTTOM":
            dist =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([waypoint_of_vehicle[0], self.height]), cv2.NORM_L2) ## Bottom
        elif projected_side == "RIGHT":
            dist =  cv2.norm(np.array(waypoint_of_vehicle) - np.array([self.width, waypoint_of_vehicle[1]]), cv2.NORM_L2) ## Right
        else:
            None
        return dist
        
        
    def distanceFromInitialVehicle(self, v_color, initial_vehicle):
        dist_from_pivot = list()
        for v_id in self.vehicles[v_color]["vehicle_info"]:
            init_vehicle_center = initial_vehicle[2]['center_of_car']
            vehicle_center = self.vehicles[v_color]["vehicle_info"][v_id]["center_of_car"]
            # print("initial vehicle center", init_vehicle_center)
            # print("vehicle_center", vehicle_center)
            dist = cv2.norm(np.array(vehicle_center) - np.array(init_vehicle_center), cv2.NORM_L2)
            dist_from_pivot.append([dist, v_id, self.vehicles[v_color]["vehicle_info"][v_id]])
        dist_from_pivot = np.array(dist_from_pivot)
        dist_from_pivot = dist_from_pivot[dist_from_pivot[:, 0].argsort()].tolist()
        return dist_from_pivot

    def extractSideDistanceOfVehicles(self, v_color, projected_side, orien_veh_ids):
        project_dist = list()
        dist_from_pivot = list()
        if orien_veh_ids:
            for v_id in self.vehicles[v_color]["vehicle_info"].keys():
                dist    = self.sideDistances(v_color, v_id, projected_side)
                project_dist.append([dist, v_id, self.vehicles[v_color]["vehicle_info"][v_id]])
            project_dist    = np.array(project_dist)
            initial_vehicle = project_dist[project_dist[:, 0].argsort()][-1]
            dist_from_pivot = self.distanceFromInitialVehicle(v_color, initial_vehicle)
            
        else:
            for v_id in self.vehicles[v_color]["vehicle_info"]:
                dist = self.sideDistances(v_color, v_id, projected_side)
                dist_from_pivot.append([dist, v_id, self.vehicles[v_color]["vehicle_info"][v_id]])
        return dist_from_pivot
        

    def generateSequenceOfMovements(self, image, time_efficiency):
        """ Extracting the sequence of movements of vehicles and aligning them from start to crash point and beyond """
        
        h, w = image.shape[:2]
        range_w = list(range(0, w))
        range_h = list(range(0, h))

        ## The frame of the image is splitted into section and used to create a point of reference to calculate the sequence of movement
        h_top = [0] * len(range_w)
        TOP = list(zip(range_w, h_top))

        h_bottom = [h] * len(range_w)
        BOTTOM = list(zip(range_w, h_bottom))

        w_left = [0] * len(range_h)
        LEFT = list(zip(w_left, range_h))

        w_right = [w] * len(range_h)
        RIGHT = list(zip(w_right, range_h))

        enviro_frame = {"TOP": TOP,
                        "BOTTOM": BOTTOM,
                        "LEFT": LEFT,
                        "RIGHT": RIGHT}

        t0 = time.time()
        
        print("\n--------- Extracting the Sequence of Movements of vehicles ----------")
        
        for v_color in self.vehicles:
            vehicle_dist_pivot = list()
            print("\n------ {} Vehicle -----".format(str.capitalize(v_color)))
            ### Least distant vehicle from the frame of the image
            least_distant, side_dist_calc = self.extractLeastDistanceVehicle(v_color)
            # print("side_dist_calc", side_dist_calc)
            vehicles_projection = list()
            for side in enviro_frame.items():
                orient_count, oriented_vehicles = self.extractVehicleProjectedSide(v_color, side)
                vehicles_projection.append([orient_count, side[0], oriented_vehicles])
            vehicles_projection = np.array(vehicles_projection)
            veh_proj_counts  = vehicles_projection[vehicles_projection[:, 0].argsort()][-2:]
            
            ## last vehicles projected side snapshot count is greater than the second last vehicle, if not then switch to distance based calculation. 
            if veh_proj_counts[-1][0] > veh_proj_counts[-2][0]: 
                veh_proj_side   = vehicles_projection[vehicles_projection[:, 0].argsort()][-1][1]
                print("Final Projected Side = ", veh_proj_side)
                print("oriented vehicles", vehicles_projection) 
                # proj_dist_veh = np.array(self.extractSideDistanceOfVehicles(
                #     v_color, veh_proj_side, vehicles_projection[vehicles_projection[:, 0].argsort()][-1][2]))
                proj_dist_veh   = np.array(self.extractSideDistanceOfVehicles(v_color, veh_proj_side, True))
                
            else:
                veh_proj_side   = str.upper(self.vehicles[v_color]["vehicle_info"][least_distant]["distance_from_boundary"][1])
                proj_dist_veh   = np.array(self.extractSideDistanceOfVehicles(v_color, veh_proj_side, False))
                proj_dist_veh   = proj_dist_veh[proj_dist_veh[:, 0].argsort()] #[::-1]
            
            print([proj_dist_veh[:,:2]])
            print("Starting Vehicle Index (ID) = ", proj_dist_veh[0][1])
            start_veh_id = proj_dist_veh[0][1] # proj_dist_veh[-1][1]
            self.vehicles[v_color]["initial_vehicle"] = self.vehicles[v_color]["vehicle_info"][start_veh_id]
            init_veh_center = self.vehicles[v_color]["initial_vehicle"]['center_of_car']
            cv2.circle(image, (int(init_veh_center[0]), int(init_veh_center[1]) ), radius=5, color=(255, 255, 0), thickness=-1)
            self.settingVehiclesInfo(v_color, proj_dist_veh.tolist())
    
        # print(proj_dist_veh)
        
        t1 = time.time()

        ### Annotating the boxpoints and the arranging the sequences
        for vehicle_color in self.vehicles:
            for i, vehicle_id in enumerate(self.vehicles[vehicle_color]["vehicle_info"]):
                box = self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][:4]
                # print("angle of the car ", self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["angle_of_car"])
                center_x, center_y = ((box[0][0] + box[1][0] + box[2][0] + box[3][0]) // 4, (box[0][1] + box[1][1] + box[2][1] + box[3][1]) // 4)
                # cv2.circle(image, (center_x, center_y), radius = 12 - 2*i , color = (255, 255, 0), thickness = -1)
                cv2.putText(image, str(i), (center_x - 5, center_y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                cv2.polylines(image, np.array([box]), True, (0, 255, 255), 2)
                if self.show_image:
                    self.pre_process.showImage("Sequence of Movements", image, time=600)
        
        # cv2.imshow("Sequence of Movements", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(self.output_folder + "{}_sequence_of_movements.jpg".format(self.process_number), image)
        self.process_number += 1
        
        time_efficiency["seq_movement"] = t1-t0
        # print("total time taken", t1-t0)
        

        
    def extractCrashImpactNodes(self, image, time_efficiency):
        """ Twelve box-point crash impact model for locating the crash point on the vehicle and the point of deformation"""
        
        t0 = time.time()
        # print("\n--------- Extracting The Crash Impact Nodes Of vehicles----------\n")

        for vehicle_color in self.vehicles:
            for vehicle_id in self.vehicles[vehicle_color]["vehicle_info"]:
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"] = dict()
                angle = self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["angle_of_car"]
                # print("vehicle color",vehicle_color, "vehicle_id", vehicle_id)
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["crashed"] = False

                ### Using the angle and the position of triangle to localize longest and shortest sides of the vehicles
                pivot_node = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][0], np.float32)
                vehicle_box_points = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][:4], np.float32)
                vehicle_side_points = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][4:8], np.float32)
                vehicle_extreme_points = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["vehicle_nodes"][8:], np.float32)
                triangle_position  = np.asarray(self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["triangle_position"], np.float32)
                front_nodes = self.pointNearNodes(vehicle_box_points, triangle_position)[:2]
                front_nodes = front_nodes[:,1:]

                front_side = self.pointNearNodes(front_nodes, pivot_node)
                front_side = front_side[:,1:]

                interval_one = list(range(0, 91)) + list(range(271, 361))
                interval_two = list(range(91, 271))

                if (round(angle) in interval_one):
                    front_left  = front_side[1]  ### longer_side ... Left_side
                    front_right = front_side[0]  ### shorter_side ... Right_side
                    # print("angle greater than 270 and less than 90", angle) 
                    
                elif (round(angle) in interval_two):
                    front_left  = front_side[0]  ### shorter_side ... Left_side
                    front_right = front_side[1]  ### longer_side  ... Right_side
                    # print("angle less than 270 and greater than 90", angle)
                    
                else:
                    None


                # if (270 <= angle or angle <= 90):
                #     front_left  = front_side[1]  ### longer_side ... Left_side
                #     front_right = front_side[0]  ### shorter_side ... Right_side
                #     print("angle greater than 270 and less than 90", angle)
                #     # print("front side", front_side)
                #     # print("front_left nodes", front_left)
                #     # print("front_right nodes", front_right)
                    
                # elif (270 > angle or angle >= 91):
                #     front_left  = front_side[0]  ### longer_side ... Left_side
                #     front_right = front_side[1]  ### shorter_side ... Right_side
                #     print("angle less than 270 and greater than 90", angle)
                #     # print("front side", front_side)
                #     # print("front_left nodes", front_left)
                #     # print("front_right nodes", front_right)

                # cv2.circle(image, tuple([int(front_left[0]),  int(front_left[1])]), 8, (255, 255, 0), -1)
                # cv2.circle(image, tuple([int(front_right[0]), int(front_right[1])]), 8, (0, 255, 255), -1)

                vehicle_corners = self.pointNearNodes(vehicle_box_points, front_left)[1:]
                vehicle_corners = vehicle_corners[:,1:]

                rear_left   = vehicle_corners[1]
                rear_right  = vehicle_corners[2]

                vehicle_sides = self.pointNearNodes(vehicle_side_points, front_left)
                vehicle_sides = vehicle_sides[:,1:]

                front_mid = vehicle_sides[0]
                left_mid  = vehicle_sides[1]
                right_mid = vehicle_sides[2]
                rear_mid  = vehicle_sides[3]

                vehicle_extremes = self.pointNearNodes(vehicle_extreme_points, front_left)
                vehicle_extremes = vehicle_extremes[:,1:]

                front_left_mid = vehicle_extremes[0]
                front_right_mid  = vehicle_extremes[1]
                rear_left_mid = vehicle_extremes[2]
                rear_right_mid  = vehicle_extremes[3]
                
                cv2.circle(image, tuple([int(front_left[0]), int(front_left[1])]), 9, (128, 128, 256), -1)

                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["front_left"]         = front_left.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["front_left_mid"]     = front_left_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["left_mid"]           = left_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["rear_left_mid"]      = rear_left_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["rear_left"]          = rear_left.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["rear_mid"]           = rear_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["rear_right"]         = rear_right.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["rear_right_mid"]     = rear_right_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["right_mid"]          = right_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["front_right_mid"]    = front_right_mid.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["front_right"]        = front_right.tolist()
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]["front_mid"]          = front_mid.tolist()
                
                self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes_array"] = [["front_left", front_left.tolist()], 
                                                                                                    ["front_left_mid", front_left_mid.tolist()], 
                                                                                                    ["left_mid", left_mid.tolist()], 
                                                                                                    ["rear_left_mid", rear_left_mid.tolist()], 
                                                                                                    ["rear_left", rear_left.tolist()], 
                                                                                                    ["rear_mid", rear_mid.tolist()], 
                                                                                                    ["rear_right", rear_right.tolist()], 
                                                                                                    ["rear_right_mid", rear_right_mid.tolist()], 
                                                                                                    ["right_mid", right_mid.tolist()], 
                                                                                                    ["front_right_mid", front_right_mid.tolist()], 
                                                                                                    ["front_right", front_right.tolist()], 
                                                                                                    ["front_mid", front_mid.tolist()]]

        t1 = time.time()

        for vehicle_color in self.vehicles:
            for vehicle_id in self.vehicles[vehicle_color]["vehicle_info"]:
                for oriented_node in self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"]:
                    point = self.vehicles[vehicle_color]["vehicle_info"][vehicle_id]["oriented_nodes"][oriented_node]
                    cv2.circle(image, tuple([int(point[0]), int(point[1])]), 4, (0, 255, 0), -1)
                    if self.show_image:
                        cv2.imshow("vehicle sides", image)
                        cv2.waitKey(15)
        if self.show_image:
            self.pre_process.showImage("vehicle sides", image, time=800)
        
        cv2.destroyAllWindows()
        cv2.imwrite(self.output_folder + "{}_twelve_point_model_sides.jpg".format(self.process_number), image)
        self.process_number += 1
        
        time_efficiency["oriented_nodes"] = t1-t0
        # print("time taken = ", t1-t0)

    
    
    def extractCrashPointOnVehicle(self, impact_image, time_efficiency, external, external_impact_points, crash_impact_locations):
        """ Crash Impact of the Individual Vehicles """
        # impact_points = list()
        t0 = time.time()
        for i, vehicle_color in enumerate(self.vehicles):
            print("\n------- Proposed Crash Impact Point on the {} Vehicle ------".format(str.capitalize(vehicle_color)))
            self.vehicles[vehicle_color]["impact_point_details"] = dict()
            min_dist = self.vehicles[vehicle_color]["crash_point"]["dist_to_vehicle"]
            impact_points = {0 : "front_left", 1: "front_left_mid", 2: "left_mid", 
                            3: "rear_left_mid", 4: "rear_left", 5: "rear_mid", 
                            6: "rear_right", 7: "rear_right_mid", 8: "right_mid", 
                            9 : "front_right_mid", 10: "front_right", 11: "front_mid"}
            
            for vehicle_vid in list(self.vehicles[vehicle_color]["vehicle_info"]): #[-2:]:
                oriented_nodes_array    = np.array(self.vehicles[vehicle_color]["vehicle_info"][vehicle_vid]["oriented_nodes_array"], dtype=object)
                crash_point             = self.vehicles[vehicle_color]["crash_point"]["coordinates"]
                oriented_nodes_label    = np.arange(0, len(oriented_nodes_array[:, 0].tolist()))
                oriented_nodes_label    = oriented_nodes_label.reshape(oriented_nodes_label.shape[0], -1)
                oriented_nodes_values   = np.asarray(oriented_nodes_array[:, 1].tolist())
                
                edist_crash_nodes       = np.sqrt(((oriented_nodes_values[:, 0] -  crash_point[0]) ** 2 + (oriented_nodes_values[:, 1] - crash_point[1]) ** 2))
                edist_crash_nodes       = edist_crash_nodes.reshape(edist_crash_nodes.shape[0], -1)
                edist_crash_nodes       = np.hstack([edist_crash_nodes, oriented_nodes_label])
                
                first_impact_side, second_impact_side  = edist_crash_nodes[edist_crash_nodes[:, 0].argsort()][:2]
                    
                if float(first_impact_side[0]) <= min_dist:
                    vehicle_impact_side = impact_points[int(first_impact_side[1])]
                    second_impact_side  = impact_points[int(second_impact_side[1])]
                    vehicle_side_coord  = oriented_nodes_values[int(first_impact_side[1])]
                    # print("\n")
                    # print(str.capitalize(vehicle_color), "Vehicle")
                    print("Snapshot of vehicle  = ", vehicle_vid)
                    print("Minimum distance from crash point to vehicle = {} pixels".format(min_dist))
                    print("Vehicle impact side internal annotation  = ", vehicle_impact_side)
                    print("Vehicle second nearest impact side       = ", second_impact_side)
                    
                    ### Adjustment to the internal annotator sides
                    if vehicle_impact_side == 'front_left_mid':
                        if second_impact_side == "front_left":
                            vehicle_impact_side = "front_left"
                        else:
                            vehicle_impact_side = "left_mid"

                    elif vehicle_impact_side == 'rear_left_mid':
                        if second_impact_side == "rear_left":
                            vehicle_impact_side = "rear_left"
                        else:
                            vehicle_impact_side = "left_mid"

                    elif vehicle_impact_side == 'rear_right_mid':
                        if second_impact_side == "rear_right":
                            vehicle_impact_side = "rear_right"
                        else:
                            vehicle_impact_side = "right_mid"

                    elif vehicle_impact_side == 'front_right_mid':
                        if second_impact_side == "front_right":
                            vehicle_impact_side = "front_right"
                        else:
                            vehicle_impact_side = "right_mid"

                    else:
                        None
                    
                    
                    self.vehicles[vehicle_color]["impact_point_details"]["snapshot"] = vehicle_vid
                    self.vehicles[vehicle_color]["impact_point_details"]["internal_impact_side"] = vehicle_impact_side
                    if external:
                        self.vehicles[vehicle_color]["impact_point_details"]["external_impact_side"] = external_impact_points[vehicle_color]
                        print("Vehicle impact side external validity = ", external_impact_points[vehicle_color])
                    self.vehicles[vehicle_color]["impact_point_details"]["side_coordinates"] = vehicle_side_coord.tolist()
                    self.vehicles[vehicle_color]["impact_point_details"]["reference_deformations"] = crash_impact_locations[vehicle_impact_side]

                    self.vehicles[vehicle_color]["vehicle_info"][vehicle_vid]["crashed"] = True
                    # impact_points.append([vehicle_color, vehicle_vid, vehicle_side, vehicle_side_coord])
                    print("Vehicle impact side internal adjusted = ", vehicle_impact_side)
                    print("Vehicle side coordinates = ", vehicle_side_coord)
                    print("Possible reference deformation group = ", self.vehicles[vehicle_color]["impact_point_details"]["reference_deformations"])
                    color = [(0, 255, 255), (255, 255, 0), (51, 255, 128), (128, 55, 160)]
                    cv2.circle(impact_image, tuple([int(vehicle_side_coord[0]), int(vehicle_side_coord[1])]), 6, color[i], -1)
                    # cv2.imshow("impact point on the vehicles", impact_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()


        t1 = time.time()

        if self.show_image:
            self.pre_process.showImage("impact point on the vehicles", impact_image, time=800)

        cv2.imwrite(self.output_folder + "{}_crash_point_on_vehicles.jpg".format(self.process_number), impact_image)
        self.process_number += 1
        
        time_efficiency["skt_veh_impact"] = t1-t0
        # print("\ntime taken",  t1-t0)
    

    def setColorBoundary(self, red_boundary, blue_boundary):
        self.pre_process.red_car_boundary = red_boundary
        self.pre_process.blue_car_boundary = blue_boundary

    def extractVehicleInformation(self, image_path, time_efficiency, show_image, output_folder, external, external_impact_points, crash_impact_locations, car_length_sim):
        image = self.pre_process.readImage(image_path=image_path)
        print("Image Dimensions", image.shape[:2])
        self.height, self.width = image.shape[:2]
        self.car_length_sim = car_length_sim
        self.show_image     = show_image        
        self.output_folder  = os.path.join(output_folder, "car/")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # self.output_folder = os.path.join(output_folder, "car/")
        self.process_number = 0
        
        """ Resize the image """
        # image = self.pre_process.resize(image=image)
        # print("Image Dimensions after resizing", image.shape[:2])
        
        """ Get Mask for the Image dimension  """
        mask = self.pre_process.getMask(image=image)
        self.pre_process.showImage('image original', image, time=800)
        
        """ Transform image HSV colorspace and threshold the image"""
        hsv = self.pre_process.changeColorSpace(image, cv2.COLOR_BGR2HSV)
        mask_r, mask_b = self.pre_process.getMaskWithRange(image=hsv)
        result_r = self.pre_process.bitwiseAndOperation(image=image, mask=mask_r)
        result_b = self.pre_process.bitwiseAndOperation(image=image, mask=mask_b)
        mask_result_r = np.hstack([cv2.merge([mask_r, mask_r, mask_r]), result_r])
        mask_result_b = np.hstack([cv2.merge([mask_b, mask_b, mask_b]), result_b])
        blend_masks_r_b = self.pre_process.bitwiseOrOperation(mask_b, mask_r)
        blend_masks_res = self.pre_process.bitwiseOrOperation(result_b, result_r)
        
        """ Blurring the masks of the red and blue cars """
        mask_r_blur = self.pre_process.blurImage(mask_r, (5, 5), 0)  # (5, 5)
        mask_b_blur = self.pre_process.blurImage(mask_b, (3, 3), 0)  # (3, 3)

        """ Morphological Operations """
        opening_r = self.pre_process.applyMorphologicalOperation(
            image=mask_r_blur, kernel_window=(5, 5), morph_operation=cv2.MORPH_OPEN)
        opening_b = self.pre_process.applyMorphologicalOperation(
            image=mask_b_blur, kernel_window=(5, 5), morph_operation=cv2.MORPH_OPEN)

        time_efficiency["preprocess"] = 0.0
        
        """ Print or show images on screen"""
        # self.pre_process.showImage('mask and result for red car', mask_result_r)
        # self.pre_process.showImage('mask and result for blue car', mask_result_b)
        # self.pre_process.showImage("masks blue and red car", blend_masks_r_b)
        # self.pre_process.showImage('result blue and red car', blend_masks_res)
        # self.pre_process.showImage("Opening Operation on Red and Blue Cars", np.hstack([opening_b, opening_r]))

        """ Plot Images"""
        # self.pre_process.plotFigure(image=mask_result_r, cmap="brg", title="Red Car")
        # self.pre_process.plotFigure(image=mask_result_b, cmap="brg", title="Blue Car")
        # self.pre_process.plotFigure(image=blend_masks_r_b, cmap="brg", title="masks blue and red car")
        # self.pre_process.plotFigure(image=blend_masks_res, cmap="brg", title="result blue and red car")
        # self.pre_process.plotFigure(image=np.hstack([opening_b, opening_r]), cmap="brg", title="Opening Operation on Red and Blue Cars")
        
        """ Saving the figure"""
        cv2.imwrite(self.output_folder + "{}_mask_result_r.jpg".format(self.process_number), mask_result_r)
        cv2.imwrite(self.output_folder + "{}_mask_result_b.jpg".format(self.process_number), mask_result_b)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_blend_masks_r_b.jpg".format(self.process_number), blend_masks_r_b)
        cv2.imwrite(self.output_folder + "{}_blend_masks_res.jpg".format(self.process_number), blend_masks_res)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_opening_morph.jpg".format(self.process_number), np.hstack([opening_b, opening_r]))
        self.process_number += 1
        
        
        self.vehicles["red"]["mask"] = opening_r
        self.vehicles["blue"]["mask"] = opening_b
        
        self.extractVehicleContoursFromMask()
        self.geometricOperationOnVehicle(image.copy(), time_efficiency)
        crash_point = self.extractingCrashPoint(image.copy(), time_efficiency)
        self.extractTriangle(image.copy(), time_efficiency)
        self.extractingAnglesForVehicles(image.copy(), time_efficiency)
        self.generateSequenceOfMovements(image.copy(), time_efficiency)
        self.extractCrashImpactNodes(image.copy(), time_efficiency)
        self.extractCrashPointOnVehicle(image.copy(), time_efficiency, external, external_impact_points, crash_impact_locations)
        
        return self.vehicles, time_efficiency
