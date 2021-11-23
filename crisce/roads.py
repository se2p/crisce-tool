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
import time
from pre_processing import Pre_Processing 
import cv2



class Roads():
    
    def __init__(self):
        self.road_parm  = True
        self.pre_process = Pre_Processing()
        self.height     = None
        self.width      = None
        self.show_image = None
        self.output_folder  = None
        self.process_number = None
        self.roads      = dict()
        
        
    def extractContours(self, morph_img, road_image, car_length, car_width):
        contours, hierarchy = cv2.findContours(morph_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        small_contours = []
        large_contours = []
        image = road_image.copy()
        # print(len(contours), hierarchy)
        for i, contour in enumerate(contours):
            # cnt = max(contours, key=cv2.contourArea)
            # if cv2.contourArea(contour) :
            # epsilon = 0.01*cv2.arcLength(contour, False)
            # approx = cv2.approxPolyDP(contour, epsilon, False)
            if cv2.arcLength(contour, False) < car_length * 2.5:
                small_contours.append(contour)
            else:
                contour = contour.reshape(contour.shape[0], 2)
                large_contours.append(contour)

            cv2.drawContours(image, [contour], 0, (255, 0, 0), -1)
            # #  cv2.polylines(image, [approx], False, (0, 255, 0), 2)
            # print("contour # {},  Shape = {},  Area = {},  Arc_Length = {} ".format(i, contour.shape, cv2.contourArea(contour), cv2.arcLength(contour, closed=False)))
        if self.show_image:
            self.pre_process.showImage("Contours Visualization", image, time=500)
        
        cv2.imwrite(self.output_folder + "{}_Contour_Viz_image.jpg".format(self.process_number), image)
        self.process_number += 1
            
        print(" large contours = {},   small contours = {} ".format(len(large_contours), len(small_contours)))
        
        return small_contours, large_contours


    def extractLengthOfRoads(self, lane_midpoints):
        """ Calculating the arc length of road midpoints of the lane"""
        lenght_of_lanes = list()
        for road in lane_midpoints:
            road = np.array(road)
            # npts = len(road) 
            x = road[:, 0]
            y = road[:, 1]
            arc = 0
            for k in range(0, len(road)-1): # or  road.shape[0]
                arc = arc + np.sqrt((x[k+1] - x[k])**2 + (y[k+1] - y[k])**2)
            lenght_of_lanes.append(arc)

        return lenght_of_lanes

    
    # Midpoints of the Lane Calculation
    def midpointOfTheLane(self, image, sample_size, lane_contour):
        print("\n")
        print("----- Extracting the Midpoints of the Lane -----")
        print("\n")
        midpoint_of_lane = []
        values = []
        euc_dist_bet_lanes = list()
        width_of_lane = list()
        max_road_point = (min(lane_contour[0].shape[0], lane_contour[1].shape[0]))
        print("max_road_point for lane", max_road_point)
        for i in range(0, max_road_point, sample_size):
            ###  Midpoint between two points = (X2+X1)^2 / (Y2+Y1)^2)
            # print("midpoints :", midpoint)
            midpoint = [(lane_contour[1][i][0] + lane_contour[0][i][0]) / 2,
                        (lane_contour[1][i][1] + lane_contour[0][i][1]) / 2]
            midpoint_of_lane.append(midpoint)
            ### Lane width in pixels
            dist = cv2.norm(lane_contour[0][i] - lane_contour[1][i], cv2.NORM_L2)
            euc_dist_bet_lanes.append(dist)
            # cv2.circle(image, tuple(midpoint), 3, (0, 255, 0), -1)
            values.append(i)

        midpoint_of_lane.sort()
        midpoints_sorted = list()

        for i, point in enumerate(midpoint_of_lane):
            if(i % 2 == 0):
                midpoints_sorted.append(point)
            elif (i == len(midpoint_of_lane) - 1):
                value = midpoint_of_lane[-1]
                # midpoints_sorted.append((value[0], value[1] - 0)
                midpoints_sorted.append((value[0], value[1]))

        adjusted_midpoints_of_lane = self.removeRedundantMidpointsOfLane(midpoints_sorted, sample_size)
        # adjusted_midpoints_of_lane =  self.removeRedundantMidpointsOfLane(self.removeRedundantMidpointsOfLane(midpoints_sorted))
        ### Lane width
        width_of_lane.append(max(euc_dist_bet_lanes))
        ### Lane length
        length_of_lane = self.extractLengthOfRoads([adjusted_midpoints_of_lane])

        for point in adjusted_midpoints_of_lane:
            cv2.circle(image, tuple([int(point[0]), int(point[1])]), 3, (0, 255, 0), -1)
        
        cv2.imwrite(self.output_folder + "{}_midpoints_of_lane.jpg".format(self.process_number), image)
        self.process_number += 1

        return width_of_lane, length_of_lane, adjusted_midpoints_of_lane


    def midpointOfFourWayAndTSection(self, canvas, large_contours):
        print("\n")
        print("----- Extracting the Midpoints of the Lane -----")
        print("\n")
        final_midpoints_lanes = list()
        euc_dist_bet_lanes = list()
        width_of_lanes = list()
        min_euc_dist_bet_lane = list()
        ordered_canvas = canvas.copy()

        for lane_1, lane_2 in itertools.combinations(large_contours, 2):
            ed_bet_two_lanes = list()
            final_lane_dist = list()
            midpoint_of_lane = list()
            # print("lane_1", lane_1.shape)
            # print("lane_2", lane_2.shape)

            for point_lane_1 in lane_1[0::15, :]:
                temp_list = list()
                for point_lane_2 in lane_2[0::15, :]:

                    #  dist = cv2.norm(pts - dst, cv2.NORM_L2)
                    euclidean_distance = math.sqrt(
                                        math.pow((point_lane_2[0] - point_lane_1[0]), 2) +
                                        math.pow((point_lane_2[1] - point_lane_1[1]), 2))

                    temp_list.append([euclidean_distance, point_lane_1.tolist(), point_lane_2.tolist()])

                ed_bet_two_lanes.append(min(temp_list))

            min_ed_point = min(ed_bet_two_lanes)
            min_euc_dist_bet_lane.append(min_ed_point)
            # print("min_euc_dist_bet_lane = ", min_euc_dist_bet_lane)

            # max_ed_point = max(ed_bet_two_lanes)
            # road_width_range = max_ed_point[0] - min_ed_point[0]
            # print("\n \n")
            # print("min_ed_point", min_ed_point)
            # print("max_ed_point", max_ed_point)

            for pt in ed_bet_two_lanes:
                if (pt[0] == min_ed_point[0]) or (pt[0] <= min_ed_point[0] + 6):
                    final_lane_dist.append(pt)

            if (len(final_lane_dist) < 17):  # 9
                final_lane_dist = list()
                for pt in ed_bet_two_lanes:
                    if (pt[0] == min_ed_point[0]) or (pt[0] <= min_ed_point[0] + 20):
                        final_lane_dist.append(pt)

            # final_lane_dist = sorted(final_lane_dist)
            euc_dist_bet_lanes.append(final_lane_dist)

        avg_road_width = 0
        lane_count = 0
        for i, lane in enumerate(euc_dist_bet_lanes):
            avg_road_width = avg_road_width + min(lane)[0] # avg_road_width = avg_road_width + max(lane)[0]
            lane_count += 1

        avg_road_width = avg_road_width / lane_count

        for i, final_lane_dist in enumerate(euc_dist_bet_lanes):
            ed_bet_two_lanes = list()
            midpoint_of_lane = list()
            # print("final_lane_dist", final_lane_dist)

            ## if (min_euc_dist_bet_lane[i][0] * 1.4 >= min(final_lane_dist)[0])
            ## if (min(min_euc_dist_bet_lane)[0] * 1.4 >= min(final_lane_dist)[0]):
            if (avg_road_width * 1.24 >= min(final_lane_dist)[0]):
                for ed, pt1, pt2 in final_lane_dist[:len(final_lane_dist) // 2]:
                    cv2.circle(canvas, tuple([int(pt1[0]), int(pt1[1])]), 4, (0, 0, 255), -1)
                    cv2.circle(canvas, tuple([int(pt2[0]), int(pt2[1])]), 4, (255, 0, 0), -1)
                    # print("min(min_euc_dist_bet_lane)[0] * 1.04", min_euc_dist_bet_lane[i][0] * 1.04)
                    break

                # cv2.imshow("Points on the circle", canvas)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                for ed, pt1, pt2 in final_lane_dist[:len(final_lane_dist) // 2]:
                    midpoint = [(pt1[0] + pt2[0]) // 2,
                                (pt1[1] + pt2[1]) // 2]
                    cv2.circle(canvas, tuple([round(midpoint[0]), round(midpoint[1])]), 5, (0, 255, 0), -1)
                    midpoint_of_lane.append(midpoint)

                ## removing the points that are near to each other extracted in the midpoints of a lane
                # print("len of the midpoint_of_lane", len(midpoint_of_lane))
                adjusted_midpoints_of_lane = self.removeRedundantMidpointsOfLane(midpoint_of_lane, 15) 
                # adjusted_midpoints_of_lane = self.removeRedundantMidpointsOfLane( self.removeRedundantMidpointsOfLane(midpoint_of_lane))

                for point in adjusted_midpoints_of_lane:
                    cv2.circle(canvas, tuple([round(point[0]), round(point[1])]), 2, (0, 0, 255), -1)

                final_midpoints_lanes.append(adjusted_midpoints_of_lane)
                # print("hello")
                width_of_lanes.append(min_euc_dist_bet_lane[i][0] * 1.04)

                # final_midpoints_lanes.append(midpoint_of_lane)
                # print("midpoint_of_lane = ", len(midpoint_of_lane))

                # for point in midpoint_of_lane:
                #     cv2.circle(canvas, tuple([int(point[0]), int(point[1])]), 5, (0, 255, ), -1)
            else:
                continue

        # cv2.imshow("canvas_1", canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        cv2.imwrite(self.output_folder + "{}_midpoints_of_lane.jpg".format(self.process_number), canvas)
        self.process_number += 1
        
        number_of_lanes = len(large_contours)
        if len(final_midpoints_lanes) > number_of_lanes:
            lanes_count = list()
            print(" final lane midpoint are larger in length ")
            while(1):
                for lane in final_midpoints_lanes:
                    lanes_count.append(len(lane))
                index = lanes_count.index(min(lanes_count))
                final_midpoints_lanes.pop(index)
                if len(final_midpoints_lanes) == number_of_lanes:
                    break
            return euc_dist_bet_lanes, width_of_lanes, self.orderedMidpointsOfTheLanes(final_midpoints_lanes, ordered_canvas)
        else:
            return euc_dist_bet_lanes, width_of_lanes, self.orderedMidpointsOfTheLanes(final_midpoints_lanes, ordered_canvas)

        # return euc_dist_bet_lanes, width_of_lanes, self.orderedMidpointsOfTheLanes(final_midpoints_lanes, ordered_canvas)


    def removeRedundantMidpointsOfLane(self, midpoint_of_lane, sample_size):
        ## Removing the points that are near to each other neareast neighbours in the vicnity of 10 euclidean points
        while(1):
            adjusted_midpoints_of_lane = list()
            final_road_points = list()
            adjust = False
        
            ## Removing Duplicate Lane Midpoints
            for pt in midpoint_of_lane:
                if pt not in final_road_points:
                    final_road_points.append(pt)

            # print("Redundant midpoints of lane are removed")
            check_redundancy = list()
            for i, point in enumerate(final_road_points):
                # print("neighbouring midepoint are removed as well")
                if (i != len(final_road_points) - 1):
                    # if (not adjust):
                    ref_point = np.array(final_road_points[i])
                    road_points = np.array(final_road_points)
                    # index = np.where(road_points, ref_point)
                    # road_points = np.delete(road_points, index)
                    
                    edist = np.sqrt(( (road_points[:, 0] - ref_point[0]) ** 2 + (road_points[:, 1] - ref_point[1]) ** 2))
                    edist = edist.reshape(edist.shape[0], -1)
                    road_points = road_points.reshape(road_points.shape[0], -1)
                    edist_road_points = np.hstack([edist, road_points])
                    # print("edist_last_points", edist_last_points)
                    # furthest_point =  edist_road_points[edist_last_points[:,0].argsort()][-1]
                    nearest_point =  edist_road_points[edist_road_points[:,0].argsort()][1]
                    nearest_point_dist = nearest_point.tolist()[0]
                    nearest_point_value = nearest_point.tolist()[1:]
                    # # dist = cv2.norm(np.array(point - road_points), cv2.NORM_L2)
                    # print("nearest_point = ", nearest_point)
                    # print("nearest_point = ", nearest_point_value)
                    current_point = ref_point.tolist()
                    
                    if nearest_point_dist <= (sample_size - 2) :  # 10 , 15
                        point = [(current_point[0] + nearest_point_value[0]) / 2,
                                    (current_point[1] + nearest_point_value[1]) / 2]
                        adjusted_midpoints_of_lane.append(point)
                        # print("midpoint_of_lane[i] = {} and midpoint_of_lane[i+1] = {} adjusted to be point = {}".format(final_road_points[i], final_road_points[i+1],  mid_value))
                        check_redundancy.append(1)
                        # adjust = True
                    else:
                        adjusted_midpoints_of_lane.append(point)
                        check_redundancy.append(0)
                        # print("midpoint_of_lane[i] = {}".format(final_road_points[i]))
                    # else:
                    #     None
                    #     # adjust = False
                else:
                    ### The lane has reached its end point adding the last point if its greater at distance from second last point
                    dist = cv2.norm(np.array(final_road_points[i]) - np.array(final_road_points[i-1]), cv2.NORM_L2)
                    if (dist > (sample_size - 2)):
                        adjusted_midpoints_of_lane.append(point)
                    # break
                    # print("The lane has reached its endpoint ")
            midpoint_of_lane = adjusted_midpoints_of_lane
            # print("check redundancy = ", check_redundancy)
            if sum(check_redundancy) < 1:
                break

        return midpoint_of_lane
        
        

    def orderedMidpointsOfTheLanes(self, final_midpoints_lanes, ordered_canvas):
        """
        Ordering of the midpoints of the lane
        """
        print("\n")
        print("----- Ordering the Midpoints of the 3 or 4 Legged Lane -----")
        print("\n")
        final_ordered_midpoint = list()
        for lane_midpoints in final_midpoints_lanes:
            ## creating list for accumulation of distance of lowest distance a point has towards its side for aligning and sorting them
            extreme_dist_lane_points = list()
            lane_midpoints = np.array(lane_midpoints)
            for point in lane_midpoints:

                ### creating reference points to for distance calculation between point( on the midpoint of a lane)
                ### its extreme boundary points.
                ### e.g (10, 15) x_low = (0, 15), x_high = (w, 15), y_low = (10, 0), y_high = (10, h)
                x_low = [0, point[1]]
                x_high = [self.width, point[1]]
                y_low = [point[0], 0]
                y_high = [point[0], self.height]
                ### for vectorized computation and avoiding for loops overhead
                extreme_sides = np.array([x_low, x_high, y_low, y_high])
                edist = np.sqrt(((extreme_sides[:, 0] - point[0]) ** 2 + (extreme_sides[:, 1] - point[1]) ** 2))
                ### extracting the minimum distance a point have towards its extreme four sides
                min_dist = min(edist)
                extreme_dist_lane_points.append([min_dist, point.tolist()])

            # print("lane_midpoints", lane_midpoints)
            # print("extreme_dist_lane_points = ", extreme_dist_lane_points)

            ### finding the point among the all midpoints of a lane which has lowest possible euclidean distance
            extreme_point = min(extreme_dist_lane_points)[1]
            # print("extreme_point ", extreme_point)

            ### calculating euclidean distance of a lowest point to the every other point in the midpoint of the lane
            ### reshaping the array to for horizontal stacking or merging with the lane midpoints
            ### now as we have an array e.g. like array([euclidean_distace of midpoint to the lowest point to the side, [midpoint]])
            ### using this format sort the array is acending order of the euclidean distance to the side.
            edist_lane = np.sqrt(((lane_midpoints[:, 0] - extreme_point[0]) ** 2 + (lane_midpoints[:, 1] - extreme_point[1]) ** 2)) 
            edist_lane = edist_lane.reshape(edist_lane.shape[0], -1)
            edist_lane_midpoints = np.hstack([edist_lane, lane_midpoints])
            edist_lane_midpoints = edist_lane_midpoints[edist_lane_midpoints[:, 0].argsort()]

            ### extracting the sorted point and appending them to the final_list of midpoints
            ### sorted midpoints of the lane can be vizualed on the picture.
            final_lane_midpoints = edist_lane_midpoints[:, 1:].tolist()
            final_ordered_midpoint.append(final_lane_midpoints)
            for midpoint in final_lane_midpoints:
                cv2.circle(ordered_canvas, tuple([round(midpoint[0]), round(midpoint[1])]), 5, (0, 255, 0), -1)
            #     cv2.imshow("image", ordered_canvas)
            #     cv2.waitKey(20)
            # cv2.destroyAllWindows()
        
        cv2.imwrite(self.output_folder + "{}_midpoints_in_order.jpg".format(self.process_number), ordered_canvas)
        self.process_number += 1

        return final_ordered_midpoint


    def centroidBetweenLanes(self, orderd_lane_points):
        last_points = list()
        midpoint = None
        # print("orderd_lane_points", orderd_lane_points)
        for lane in orderd_lane_points:
            last_points.append(lane[-1])
            # print("last points", lane[-1])
            # print("last_points list", last_points)
        if len(orderd_lane_points) == 3:
            midpoint = [(round(last_points[0][0]) + round(last_points[1][0]) +  round(last_points[2][0])) // len(orderd_lane_points),
                        (round(last_points[0][1]) + round(last_points[1][1]) + round(last_points[2][1])) // len(orderd_lane_points)]
        elif len(orderd_lane_points) == 4:
            midpoint = [(round(last_points[0][0]) + round(last_points[1][0]) +  round(last_points[2][0]) +
                        round(last_points[3][0])) // len(orderd_lane_points),
                        (round(last_points[0][1]) + round(last_points[1][1]) +  round(last_points[2][1]) +
                        round(last_points[3][1])) // len(orderd_lane_points)]
        else:
            midpoint = None

        return midpoint


    def getExtrapolatedPointMidpoints(self, canvas, traverse_parameter, ordered_mid_lane):

        print("\n \n ")
        print("----- Extrapolating the lane -----")
        print("\n \n ")
        ord_mid_lane = ordered_mid_lane.copy()
        centroid = self.centroidBetweenLanes(ord_mid_lane)
        cv2.circle(canvas, tuple(centroid), 5, (255, 0, 0), -1)
        extrapolated_ordered_midpoint = []

        for lane_midpoints in ordered_mid_lane:
            x_sim = [lane_midpoints[2][0] in i for i in lane_midpoints[2:6]]
            x_sim = np.array(x_sim).all()

            y_sim = [lane_midpoints[2][1] in i for i in lane_midpoints[2:6]]
            y_sim = np.array(y_sim).all()

            # print("original midpoints", lane_midpoints)

            if x_sim:
                x = lane_midpoints[-1][0]
                y = lane_midpoints[-1][1]
                change_y = centroid[1] - lane_midpoints[-1][1]
                dy = math.sqrt(math.pow(change_y, 2))
                lane_midpoints.pop()
                # print("centroid = ", centroid)
                if change_y < 0:
                    for i in range(0, int(dy * 1.3), traverse_parameter):
                        point = [x, y - i]
                        # print("point", point)
                        lane_midpoints.append(point)
                elif change_y == 0:
                    print("No change in y")
                else:
                    for i in range(0, int(dy * 1.5), traverse_parameter):
                        point = [x, y + i]
                        lane_midpoints.append(point)

            elif y_sim:
                x = lane_midpoints[-1][0]
                y = lane_midpoints[-1][1]
                change_x = centroid[0] - lane_midpoints[-1][0]
                dx = math.sqrt(math.pow(change_x, 2))
                lane_midpoints.pop()
                # print("centroid = {} and dx = {} change_x = {} ".format(centroid, dx, change_x))
                if change_x < 0:    # 1
                    for i in range(0, int(dx * 1.5), traverse_parameter):
                        point = [x - i, y]
                        lane_midpoints.append(point)
                        # print("point", point)
                elif change_x == 0:  # 2
                    print("No change in x")
                else:
                    for i in range(0, int(dx), traverse_parameter):
                        point = [x + i, y]
                        lane_midpoints.append(point)

            else:
                """
                T-section or Merge-into or Fork-into or 4-way type of  road variant
                """
                ### The Points on the T-section or 4-Way lane are not horizontal or vertical to the axis to be interpolated
                ### The points are constantly changing their x,y position so, e.g. diagnal points, curve points on T-section and 4-way.
                ###
                number_of_roads = len(ordered_mid_lane)
                current_last_point = np.array(lane_midpoints[-1])
                # print("The Road does not have constant values on the same axis either on the x or y")
                # print("\n \n ")
                reference_last_points = list()
                for lane in ordered_mid_lane:
                    reference_last_points.append(lane[-1])
                reference_last_points = np.array(reference_last_points)
                edist_last_points = np.sqrt(((reference_last_points[:, 0] - current_last_point[0]) ** 2 + (reference_last_points[:, 1] - current_last_point[1]) ** 2))
                edist_last_points = edist_last_points.reshape(edist_last_points.shape[0], -1)
                reference_last_points = reference_last_points.reshape(reference_last_points.shape[0], -1)
                edist_last_points = np.hstack([edist_last_points, reference_last_points])
                # print("edist_last_points", edist_last_points)
                furthest_point =  edist_last_points[edist_last_points[:, 0].argsort()][-1]
                neareast_point =  edist_last_points[edist_last_points[:, 0].argsort()][1]
                # print("furthest_point", furthest_point.tolist())
                # print("current_last_point", current_last_point.tolist())

                point_1 = lane_midpoints[-2]
                point_2 = current_last_point.tolist()
                x = [round(point_1[0]), round(point_2[0])]
                y = [round(point_1[1]), round(point_2[1])]
                dy = y[1] - y[0]
                dx = x[1] - x[0]

                rads = math.atan2(dy, dx)
                angle = math.degrees(rads)
                print("Angle of the lane = ", angle)
                length = neareast_point.tolist()[0]
                extrap_point_x = int(round(point_2[0] + length * 1.1 * math.cos(angle * np.pi / 180.0)))
                extrap_point_y = int(round(point_2[1] + length * 1.1 * math.sin(angle * np.pi / 180.0)))
                cv2.line(canvas, tuple([int(point_2[0]), int(point_2[1])]), tuple([extrap_point_x, extrap_point_y]), (255, 0, 0), 2)
                lane_midpoints.extend([(extrap_point_x, extrap_point_y)])

            # print("new extrapolated midpoints", lane_midpoints)
            extrapolated_ordered_midpoint.append(lane_midpoints)

        return extrapolated_ordered_midpoint


    def calculateAspectRatio(self, car_length, car_length_sim):
        return car_length_sim / car_length  # flaot values required

    def distortionMappingVizualization(self, road_image, aspect_ratio, lane_contours, number_of_roads):
        if number_of_roads == 2:
            lane_midpoints = np.array(lane_contours)
            # print("lane_midpoints = ", lane_midpoints)
            distortionMapping = lambda x, r: x * r
            result = distortionMapping(lane_midpoints, aspect_ratio)
            # result = np.array(np.asarray(lane_midpoints) * aspect_ratio, np.uint8)
            # result = np.array(np.array(lane_midpoints) * aspect_ratio)
            # print("results of distortion and mapping", result)
            # new_lane_contours.append(result)
            for pt in result:
                ## Visualizing the distorted road
                ptx = round(pt[0])
                pty = round(pt[1])
                cv2.circle(road_image, (ptx, pty), 2, (255,0,0), -1)
                
            cv2.imwrite(self.output_folder + "{}_distortion_mapping.jpg".format(self.process_number), road_image)
            self.process_number += 1
            
            return result.tolist()
        else:
            new_lane_contours = []
            for lane in lane_contours:
                distortionMapping = lambda x, r: x * r
                result = distortionMapping(lane, aspect_ratio)
                # print("results of distortion and mapping",result)
                new_lane_contours.append(result)
                for pt in result:
                    ## Visualizing the distorted road
                    ptx = round(pt[0])
                    pty = round(pt[1])
                    cv2.circle(road_image, (ptx, pty), 2, (255,0,0), -1)
                    
            cv2.imwrite(self.output_folder + "{}_distortion_mapping.jpg".format(self.process_number), road_image)
            self.process_number += 1
            
            return new_lane_contours

    def adjustRoadToSimulation(self, dist_height):
        """ Adjusting the Road for the simulation settings """
        adjusted_lane_midpoints = list()
        for lane in self.roads["small_lane_midpoints"]:
            lane = np.array(lane)
            adjusted_lane_midpoints.append(np.hstack((lane[:, 0].reshape(lane.shape[0], 1), dist_height - lane[:, 1].reshape(lane.shape[0], 1))).tolist())

        self.roads["simulation_lane_midpoints"] = adjusted_lane_midpoints
        
    def settingRoadToBeamNG(self):
        """ Setting the roads in BeamNG format """
        lane_nodes = list()
        for i, lane in enumerate(self.roads["simulation_lane_midpoints"]):
            nodes = list()
            road_lane_width = self.roads["scaled_lane_width"][i]
            for j, node in enumerate(lane):
                point = node.copy()
                # (car_length * a_ratio / 4 * 0.9) * 2 can be use istead
                point.extend([0, round(road_lane_width, 3)])
                nodes.append(tuple(point))
            lane_nodes.append(nodes)
        
        return lane_nodes


    def extractRoadInformation(self, image_path, time_efficiency, show_image, output_folder, car_length, car_width, car_length_sim):
        ## Read the image and create a blank mask
        image = self.pre_process.readImage(image_path=image_path)
        print("\n------------Road-------------\n")
        print("Image Dimensions", image.shape[:2])
        self.height, self.width = image.shape[:2]
        self.show_image     = show_image
        self.output_folder  = os.path.join(output_folder, "road/")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.process_number = 0

        """ Resize the image """
        # image = self.pre_process.resize(image=image)
        # print("Image Dimensions after resizing", image.shape[:2])
        
        """ Get Mask for the Image dimension  """
        mask = self.pre_process.getMask(image=image)

        """Transform to gray colorspace and threshold the image"""
        gray = self.pre_process.changeColorSpace(image=image, color_code=cv2.COLOR_BGR2GRAY)
        blur = self.pre_process.blurImage(image=gray, kernel_size=(3, 3), sigmaX= 0)
        thresh = self.pre_process.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        ''' Use of Horizontal and Vertical Morphological Kernal for increasing the pixel intesities along X and Y-axis'''
        # dilate_kernel = np.ones((10, 10), np.uint8)  # note this is a horizontal kernel
        dilate_image = self.pre_process.dilate(thresh, kernel_window=(10, 10))
        erode_image = self.pre_process.erode(dilate_image, kernel_window=(10, 10))

        """ Morphological Operations """
        morph_img = erode_image.copy()
        ## rect_kernel = cv2.getStructuringElement( cv2.MORPH_RECT,(10,10))
        ## ellipse_kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(15,15))
        ## morph_img = self.pre_process.applyMorphologicalOperation(thresh, cv2.MORPH_CLOSE, ellipse_kernel)
        ## kernel = np.ones((20,20),np.uint8)
        # opening = self.pre_process.applyMorphologicalOperation(thresh, cv2.MORPH_OPEN, kernel_window=(20,20))

        """ Show and Plot images """
        # self.pre_process.showImage('image original', image)
        # self.pre_process.showImage(title="thresholded image", image=thresh)
        # self.pre_process.showImage("dilate", dilate_image)
        # self.pre_process.showImage("erode", erode_image)
        # self.pre_process.showImage("Morphological Close Operation ", erode_image)
        # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
        # self.pre_process.plotFigure(thresh)
        # self.pre_process.plotFigure(morph_img, cmap="brg", title="Morphological Close Operation ")
        
        """ Saving the figure"""
        cv2.imwrite(self.output_folder + "{}_gray_image.jpg".format(self.process_number), gray)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_blur_image.jpg".format(self.process_number), blur)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_threshold_image.jpg".format(self.process_number), thresh)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_dilate_image.jpg".format(self.process_number), dilate_image)
        self.process_number += 1
        cv2.imwrite(self.output_folder + "{}_erode_image.jpg".format(self.process_number), erode_image)
        self.process_number += 1

        
        road_image = image.copy()
        
        small_contours, large_contours = self.extractContours(morph_img, road_image, car_length, car_width)  
        canvas = cv2.merge((morph_img, morph_img, morph_img))
        canvas = canvas.copy()
        canvas = cv2.bitwise_not(canvas)

        """
        if length of car in simulation that is 4m is divided by the length of the car in the sketch then the road cannot be curved as majority of the points will
        be omitted by the system in order to approximate the area as shown in the sketch to to the simulation.
        e.g car_length_simulation / car_length_sketch :
                4 / 82 = 0.048
                0.048 mutiplied by all pixel values to get a scale down version of the map that conforms to the simulation in sketch
                will affect the curve road by making it straight and whereas the straight road will have no affect at all. 
        """

        t0 = time.time()

        if (len(large_contours) == 2):
            ## its a one road or lane with seperation
            print("\nStraight or Curve road")
            number_of_lanes = len(large_contours)
            width_of_lanes, length_of_lanes, large_lane_midpoints = self.midpointOfTheLane(road_image, 
                                                                                            sample_size=round(car_length) // 4, 
                                                                                            lane_contour=large_contours)
            # self.pre_process.showImage("drawing contours", road_image)

            a_ratio = self.calculateAspectRatio(car_length=car_length, car_length_sim=car_length_sim)
            small_lane_midpoints = self.distortionMappingVizualization(aspect_ratio=a_ratio, 
                                                                        road_image=road_image,
                                                                        lane_contours=large_lane_midpoints,
                                                                        number_of_roads=len(large_contours))  # a_ratio / 2
            small_lane_midpoints = [small_lane_midpoints]
            scaled_lane_width = [width_of_lanes[0] * a_ratio]
            scaled_lane_length = [length_of_lanes[0] * a_ratio]

            self.roads["sketch_lane_width"]     = width_of_lanes
            self.roads["sketch_lane_length"]    = length_of_lanes
            self.roads["large_lane_midpoints"]  = large_lane_midpoints
            self.roads["small_lane_midpoints"]  = small_lane_midpoints
            self.roads["scaled_lane_width"]     = scaled_lane_width
            self.roads["scaled_lane_length"]    = scaled_lane_length

            # self.pre_process.showImage("distortion and mapping", road_image)

        elif(len(large_contours) == 3 or len(large_contours) == 4):
            ## Its T-Section road or two roads or lane with seperation
            print("\nT-Section road or Four way road")
            number_of_lanes = len(large_contours)
            ed_dist_bet_lanes, width_of_lanes, ordered_midpoints_lanes = self.midpointOfFourWayAndTSection(canvas, large_contours)
            # self.pre_process.showImage("drawing contours", road_image)

            ### If its a double straight road and not a four-way or merge-into
            try:
                extrapolated_ordered_midpoint = self.getExtrapolatedPointMidpoints(canvas=road_image, 
                                                                                   traverse_parameter=round(car_length) // 3, 
                                                                                   ordered_mid_lane=ordered_midpoints_lanes)
            except:
                extrapolated_ordered_midpoint = ordered_midpoints_lanes
                # length_of_lanes = extractLengthOfRoads(extrapolated_ordered_midpoint)
            # else:
            #     print("Nothing went wrong")

            extrap_ord_mid_lanes = list()
            small_lane_midpoints = list()
            length_of_lanes = self.extractLengthOfRoads(extrapolated_ordered_midpoint)
            for lane_midpoints in extrapolated_ordered_midpoint:
                extrap_ord_mid_lanes.append(np.array(lane_midpoints, dtype="float32"))

            a_ratio = self.calculateAspectRatio(car_length=car_length, car_length_sim=car_length_sim)
            small_lane_contours = self.distortionMappingVizualization(aspect_ratio=a_ratio, 
                                                                      road_image=road_image,
                                                                      lane_contours=large_contours,
                                                                      number_of_roads=len(large_contours))

            small_lane_mids     = self.distortionMappingVizualization(aspect_ratio=a_ratio,
                                                                      road_image=road_image,
                                                                      lane_contours=extrap_ord_mid_lanes,
                                                                      number_of_roads=len(large_contours))

            for lane_midpoints in small_lane_mids:
                small_lane = lane_midpoints.astype(np.float)
                small_lane_midpoints.append(small_lane.tolist())

            for ordered_mids in extrapolated_ordered_midpoint:
                count = 0
                for pt in ordered_mids:
                    # print( tuple([pt[0] + 5, pt[1] + 5]))
                    cv2.circle(road_image, tuple( [int(pt[0]), int(pt[1])]), 3, (0, 0, 255), -1)
                    cv2.putText(road_image, str(count), tuple( [int(pt[0]) + 15, int(pt[1]) + 10]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    # self.pre_process.showImage("canvas 3", road_image)
                    # cv2.waitKey(10)
                    count += 1
                cv2.imwrite(self.output_folder + "{}_extrapolated_ordered_midpoint.jpg".format(self.process_number), road_image)
                self.process_number += 1
                # cv2.destroyAllWindows()
                    # break

            scaled_lane_width   = [width * a_ratio for width in width_of_lanes]
            scaled_lane_length  = [length * a_ratio for length in length_of_lanes]

            self.roads["large_lane_midpoints"]  = extrapolated_ordered_midpoint
            self.roads["sketch_lane_width"]     = width_of_lanes
            self.roads["sketch_lane_length"]    = length_of_lanes
            self.roads["small_lane_midpoints"]  = small_lane_midpoints
            self.roads["scaled_lane_width"]     = scaled_lane_width
            self.roads["scaled_lane_length"]    = scaled_lane_length
            self.roads["sequence_of_lanes"]     = list(zip(small_lane_midpoints, scaled_lane_length, scaled_lane_width))

            # self.pre_process.showImage("distortion and mapping", road_image)

        t1 = time.time()
        time_efficiency["road_ext"] = t1-t0
        # print("total time = ", t1-t0)
        
        if self.show_image:
            self.pre_process.showImage("Final Road with Distortion and Mapping", road_image, time=1000)
        cv2.imwrite(self.output_folder + "{}_final_result.jpg".format(self.process_number), road_image)
        self.process_number += 1
        # self.pre_process.plotFigure(road_image, cmap="brg", title="Final Road with Distortion and Mapping")
        # self.pre_process.saveFigure('road_distorted.jpg', dpi=300)
        
        distorted_height = road_image.shape[0] * (car_length_sim / car_length)
        
        self.adjustRoadToSimulation(distorted_height)
        final_lane_nodes =  self.settingRoadToBeamNG()
        
        return self.roads, final_lane_nodes

        
        
        
