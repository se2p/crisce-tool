from pre_processing import Pre_Processing
from roads import Roads
from car import Car
from shapely.geometry import MultiLineString, Polygon
import copy
import itertools
import seaborn as sns
import pandas as pd
import time
import glob
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import numpy as np
import cv2

from bezier_path import calc_4points_bezier_path, calc_bezier_path, bernstein_poly, bezier, bezier_derivatives_control_points, curvature, plot_arrow
from cubic_spline_planner import Spline2D
from bspline_path import approximate_b_spline_path,  interpolate_b_spline_path
import scipy.interpolate as scipy_interpolate

show_animation = True

class Kinematics():
    
    def __init__(self):
        self.time_efficiency= None
        self.pre_process    = Pre_Processing()
        self.vehicles       = None
        self.height         = None
        self.width          = None
        self.show_image     = None
        self.output_folder  = None
        self.process_number = None


    def selectionVehicleSnapshots(self):
        t0 = time.time()
        """ Selection of Vehicle Snapshots for the trajectory in the simulation """
        for vehicle_color in self.vehicles:
            self.vehicles[vehicle_color]["snapshots"] = dict()
            self.vehicles[vehicle_color]["trajectories"] = dict()
            self.vehicles[vehicle_color]["trajectories"]["computed"] = dict()
            available_snapshots = len(self.vehicles[vehicle_color]["vehicle_info"].keys())
            vehicle_snapshots = list()
            mode = None
            num_snaps = None
            if(available_snapshots > 1):
                print("\nMode of snapshots to select for vehicle 0: All , 1: Manual, 2: Random = {}:".format(vehicle_color))
                print("Vehicle", vehicle_color)
                print("\t 0: All            --> Use all available snapshots for creating the trajectory (Standard) \n")
                print("\t 1: To Crash Point --> selection of all the snapshots till crash point. (Standard) \n")
                print("\t 2: Manual         --> selection or removal of any snapshot for creating the trajectory except the impact one. \n")
                print("\t 3: Random         --> selection of snapshot using random algorithm between first and last(impact) one.\n")

                while(1):
                    """Enter Mode of Snapshots: """
                    # mode = int(input("Enter Mode of Snapshots: "))
                    mode = 1
                    if (mode not in [0, 1, 2, 3]):
                        print("enter mode")
                        continue
                    else:
                        break

                if(mode == 0):
                    for v_id in self.vehicles[vehicle_color]["vehicle_info"]:
                        vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])

                elif(mode == 1):
                    print("\n mode 1 selected: selecting all the snapshots till crash point")
                    for v_id in self.vehicles[vehicle_color]["vehicle_info"]:
                        crash_vehicle = self.vehicles[vehicle_color]["vehicle_info"][v_id]["crashed"]
                        if crash_vehicle:
                            vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])
                            break
                        else:
                            vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])
                            continue

                elif(mode == 2):
                    print("\t Mode 1: Manual is selected \n")
                    while(1):
                        print("Enter the number of snapshots to select from ",available_snapshots - 1, " with a comma:")
                        print("Example: from 0 till number of the sanpshots mentioned")
                        print("\t 0,1,2,3,4")
                        num_snaps = input("Enter Number of Snapshots to Select with a comma: ")

                        """ Textual Preprocessing """
                        num_snaps = " ".join(snap for snap in num_snaps if snap.isalnum())
                        num_snaps = list(set(" ".join(snap for snap in num_snaps if snap.isdigit()).split(" ")))
                        num_snaps = sorted([int(snap) for snap in num_snaps])
                        print(num_snaps)
                        if (not set(num_snaps).issubset(np.arange(0, available_snapshots).tolist())):
                            print("Selected snapshots are not in range, please enter again. ")
                            continue
                        else:
                            break
                    for v_id in self.vehicles[vehicle_color]["vehicle_info"]:
                        if (int(v_id) in num_snaps):
                            vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])
                        else:
                            continue
                    ### appending the impact snapshot of the vehicle
                    # last_snap = list(self.vehicles["red"]["vehicle_info"].keys())[-1]
                    # vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][last_snap])

                elif(mode == 3):
                    ### first snapshot
                    vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"]["0"])
                    snapshots = np.arange(1, available_snapshots - 1).tolist()
                    number_of_random_snaps = len(snapshots) - 1
                    if(number_of_random_snaps == 0):
                        number_of_random_snaps = 1
                    num_snaps = np.random.randint(1, len(snapshots), number_of_random_snaps)
                    num_snaps = sorted(num_snaps)
                    for v_id in self.vehicles[vehicle_color]["vehicle_info"]:
                        if (int(v_id) in num_snaps):
                            vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])
                        else:
                            continue

                    ### appending the impact snapshot of the vehicle
                    last_snap = list(self.vehicles["red"]["vehicle_info"].keys())[-1]
                    vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][last_snap])

            else:
                for v_id in self.vehicles[vehicle_color]["vehicle_info"]:
                    vehicle_snapshots.append(self.vehicles[vehicle_color]["vehicle_info"][v_id])
                    # print(vehicle_color, v_id)
            ### Saving the snapshots in a data structure
            self.vehicles[vehicle_color]["snapshots"] = vehicle_snapshots
            
        t1 = time.time()
        self.time_efficiency["ext_snapshots"] = t1-t0
        # print("total time = ", t1-t0)

    def organizeWaypointForTrajectory(self, num_points, extrapolate_point):
        """ Multiple points selection as a waypoint of trajectory of the car """
        t0 = time.time()

        for vehicle_color in self.vehicles:
            snapshots = self.vehicles[vehicle_color]["snapshots"]
            waypoints = list()
            if (num_points >= 4):
                num_points = 4
            for snap in snapshots:
                tri_pos = snap["triangle_position"]

                back = snap["oriented_nodes"]["rear_mid"]
                middle = snap["center_of_car"]
                triangle = snap["triangle_position"]
                front = snap["oriented_nodes"]["front_mid"]

                if(num_points == 0):
                    waypoints.append([back, middle, front])
                elif(num_points == 1):
                    waypoints.append([middle])
                elif(num_points == 2):
                    waypoints.append([middle, front])
                elif(num_points == 3):
                    waypoints.append([back, middle, triangle])
                else:
                    print("\nNumber of vehicle waypoints options to select:  0: back, middle, triangle (Standard),  1: middle,  2: middle, triangle,  3: back, middle, front")

            ordered_waypoints = [point for points in waypoints for point in points]
            
            # print("vehicle_color: \n", np.array(ordered_waypoints))
            if extrapolate_point:
                last_point  = ordered_waypoints[-1] # last point 
                slast_point = ordered_waypoints[-2] # second last point
                x = [round(slast_point[0]), round(last_point[0])]
                y = [round(slast_point[1]), round(last_point[1])]
                dy = y[1] - y[0]
                dx = x[1] - x[0]

                rads = math.atan2(dy,dx)
                angle = math.degrees(rads)
                print("angle = ", angle)
                length = self.vehicles[vehicle_color]["dimensions"]["car_width"] * 2
                extrap_point_x =  int(round(last_point[0] + length * 1.1 * math.cos(angle * np.pi / 180.0)))
                extrap_point_y =  int(round(last_point[1] + length * 1.1 * math.sin(angle * np.pi / 180.0)))
                # cv2.line(canvas, tuple( [int(last_point[0]), int(last_point[1])]), tuple([extrap_point_x, extrap_point_y]), (255, 0, 0), 2)
                extrap_point = ((last_point[0] + extrap_point_x) // 2 , (last_point[1] + extrap_point_y) // 2 )
                ordered_waypoints.append(extrap_point)
            
            # print("vehicle_color: \n", np.array(ordered_waypoints))
            self.vehicles[vehicle_color]["ordered_waypoints"] = ordered_waypoints
        # return np.array(ordered_waypoints)
        
        # print(self.vehicles["red"]["ordered_waypoints"], self.vehicles["blue"]["ordered_waypoints"])
        
        t1 = time.time()
        self.time_efficiency["ext_waypoint"] = t1-t0
        # print("total time = ", t1-t0)
        return self.vehicles

    def selectNumberOfWaypoints(self):
        print("Number of vehicle waypoints options to select: ")
        print("\t 0:   back, middle, front  (Standard) \n")
        print("\t 1:   middle \n")
        print("\t 2:   middle, front \n")
        print("\t 3:   back, middle, triangle \n")

        num_points = None
        while(num_points == None):
            # num_points = int(input("Enter number of waypoints"))
            num_points = 0
            if (num_points not in [0, 1, 2, 3]):
                continue
            else:
                break
        self.organizeWaypointForTrajectory(num_points=int(num_points), extrapolate_point=True)


    def vehicleDistortedControlPoints(self, image):
        t0 = time.time()
        for vehicle_color in self.vehicles:
            trajectory_waypoints = np.array(self.vehicles[vehicle_color]["ordered_waypoints"])
            for pt in trajectory_waypoints:
                ## Visualizing the distorted road
                ptx = round(pt[0])
                pty = round(pt[1])
                cv2.circle(image, (ptx, pty), 4, (0, 255, 0), -1)

            # cv2.imshow("trajectory of vehicle", traj_image)
            # cv2.waitKey(0)

            distortionMapping = lambda x, r : x * r
            aspect_ratio = self.vehicles[vehicle_color]["dimensions"]["car_length_sim"] / self.vehicles[vehicle_color]["dimensions"]["car_length"]
            # aspect_ratio = self.vehicles[vehicle_color]["dimensions"]["car_width"] / self.vehicles[vehicle_color]["dimensions"]["car_length"]
            result = distortionMapping(trajectory_waypoints, aspect_ratio)  # 0.16
            for pt in result:
                ## Visualizing the distorted control point of trajecotory 
                ptx = round(pt[0])
                pty = round(pt[1])
                cv2.circle(image, (ptx, pty), 1, (0, 0, 255), -1)
                
            """ Access the distorted ordered waypoints in numpy array type"""
            # self.vehicles[vehicle_color]["distorted_ordered_waypoints"] = result
            cv2.imwrite(self.output_folder + "{}_distorted_control_points.jpg".format(self.process_number), image)
            self.process_number += 1

        t1 = time.time()
        self.time_efficiency["viz_distortion"] = t1-t0
        # print("total time = ", t1-t0)
        if self.show_image:
            self.pre_process.showImage("distorted and mapped control waypoints of vehicle", image, time=150)
        # self.pre_process.plotFigure(image=image, cmap="brg", title="distorted and mapped trajectory of vehicle", figsize=(15, 10))
        # self.pre_process.saveFigure(image_name="distort_map_traj")


    def calBezierPath(self, vehicle_color, n_points=200):
        """ Bezier Curve Trajectory Spline 2D Trajectory"""
        control_points = np.array(self.vehicles[vehicle_color]["ordered_waypoints"])
        snapshots = self.vehicles[vehicle_color]["snapshots"]
        plt.figure(figsize=(17, 7))

        if(len(snapshots) > 1):
            path = calc_bezier_path(control_points, n_points)

            # Display the tangent, normal and radius of cruvature at a given point
            t = 0.86  # Number in [0, 1]
            x_target, y_target = bezier(t, control_points)
            derivatives_cp = bezier_derivatives_control_points(control_points, 2)
            point = bezier(t, control_points)
            dt = bezier(t, derivatives_cp[1])
            ddt = bezier(t, derivatives_cp[2])
            # Radius of curvature
            radius = 1 / curvature(dt[0], dt[1], ddt[0], ddt[1])
            # Normalize derivative
            dt /= np.linalg.norm(dt, 2)
            dt = dt * 8
            tangent = np.array([point, point + dt])
            normal = np.array([point, point + [- dt[1], dt[0]]])
            curvature_center = point + np.array([- dt[1], dt[0]]) * radius
            circle = plt.Circle(tuple(curvature_center),
                                radius,
                                color=(0, 0.8, 0.8), fill=False, linewidth=1)

            # if show_animation:
            #     plt.plot(path.T[0], path.T[1], label="Bezier Path")
            #     plt.plot(control_points.T[0], control_points.T[1],'--o', label="Control Points")
            #     plt.plot(x_target, y_target)
            #     plt.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
            #     plt.plot(normal[:, 0], normal[:, 1], label="Normal")
            #     # ax.add_artist(circle)
            #     #     plot_arrow(float(control_points[0][0]), float(control_points[0][1]), start_yaw)
            #     #     plot_arrow(float(control_points[-1][0]), float(control_points[-1][1]), end_yaw)
        else:
            # plt.plot(control_points[-1][0], control_points[-1][1], label="Stationary Point")
            control_points = self.vehicles[vehicle_color]["snapshots"][0]["center_of_car"]
            plt.plot(control_points[0], control_points[1],'--o', label="Stationary Point")
            # print(control_points.T)

            path = np.array([control_points[0], control_points[1]]).reshape(1, -1)
      
        # plt.legend()
        # plt.axis("equal")
        # plt.grid(True)
        # plt.gca().invert_yaxis()
        # # fig1 = plt.gcf()
        # plt.savefig(self.output_folder + '{}_{}_vehicle_bezier_curve.jpg'.format(self.process_number, vehicle_color), dpi=150)
        # self.process_number += 1
        # # plt.show()
        # plt.close()
        
        return path


    def calBezierSplineTrajectory(self, vehicle_color, n_course_point=200):
        """ Calculating the B-Spline Trajectory """
        waypoints = np.array(self.vehicles[vehicle_color]["ordered_waypoints"])
        snapshots = self.vehicles[vehicle_color]["snapshots"]
        plt.figure(figsize=(17, 8))

        if (len(snapshots) > 1):
            # way_point_x = [-1.0, 3.0, 4.0, 2.0, 1.0]
            # way_point_y = [0.0, -3.0, 1.0, 1.0, 3.0]
            x = waypoints[:, 0].tolist()
            y = waypoints[:, 1].tolist()
            # n_course_point = 200  # sampling number
            rax, ray = approximate_b_spline_path(x, y, n_course_point)
            rix, riy = interpolate_b_spline_path(x, y, n_course_point)

            # # show results
            # # fig = plt.figure(figsize=(20,10))
            # # ax1, ax2 = fig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

            # plt.plot(x, y, '-og', label="way points")
            # plt.plot(rax, ray, '-r', label="Approximated B-Spline path")
            # # plt.plot(rix, riy, '-b', label="Interpolated B-Spline path")
            # # path = (list(zip(rax, ray)), list(zip(rix, riy)))
            path = list(zip(rax, ray))
        else:
            waypoints = self.vehicles[vehicle_color]["snapshots"][0]["center_of_car"]
            # plt.plot(waypoints[0], waypoints[1], '-og', label="Stationary Point")
            path = np.array([waypoints[0], waypoints[1]]).reshape(1, -1)

        # plt.grid(True)
        # plt.legend()
        # plt.gca().invert_yaxis()
        # plt.axis("equal")
        # # plt.savefig('{}_vehicle_bezier_spline.jpg'.format(vehicle_color), dpi=150)
        # plt.savefig(self.output_folder + '{}_{}_vehicle_bezier_spline.jpg'.format(self.process_number, vehicle_color), dpi=150)
        # self.process_number += 1
        
        
        # plt.close()

        return path

    ## TODO Remove
    def calCubicSplineTrajectory(self, vehicle_color):
        """ Calculating the Cubic Spline Trajectory """

        waypoints = np.array(self.vehicles[vehicle_color]["ordered_waypoints"])
        snapshots = self.vehicles[vehicle_color]["snapshots"]

        if (len(snapshots) > 1):

            x = waypoints[:, 0].tolist()  # [::-1]
            y = waypoints[:, 1].tolist()  # [::-1]
            ds = 2  # [m] distance of each intepolated points
            sp = Spline2D(x, y)
            s = np.arange(0, sp.s[-1], ds)

            rx, ry, ryaw, rk = [], [], [], []
            for i_s in s:
                ix, iy = sp.calc_position(i_s)
                rx.append(ix)
                ry.append(iy)
                ryaw.append(sp.calc_yaw(i_s))
                rk.append(sp.calc_curvature(i_s))

            # plt.subplots(1)
            plt.figure(figsize=(17, 7))
            plt.plot(x, y, "xb", label="input")
            plt.plot(rx, ry, "-r", label="spline")
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.gca().invert_yaxis()
            # fig1 = plt.gcf()
            # fig1.savefig('cubic_spline.jpg', dpi=150)
            # plt.savefig(self.output_folder + '{}_{}_vehicle_cubic_spline_traj.jpg'.format(self.process_number, vehicle_color), dpi=150)
            plt.close()

            # plt.subplots(1)
            plt.figure(figsize=(17, 7))
            plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
            plt.grid(True)
            plt.legend()
            plt.xlabel("line length[m]")
            plt.ylabel("yaw angle[deg]")
            # plt.savefig(self.output_folder + '{}_{}_vehicle_cubic_spline_yaw.jpg'.format(self.process_number, vehicle_color), dpi=150)
            plt.close()

            # plt.subplots(1)
            plt.figure(figsize=(17, 7))
            plt.plot(s, rk, "-r", label="curvature")
            plt.grid(True)
            plt.legend()
            plt.xlabel("line length[m]")
            plt.ylabel("curvature [1/m]")
            plt.close()

            path = (rx, ry, ryaw, rk)
            path = list(zip(rx, ry))

        else:
            plt.figure(figsize=(17, 7))
            # path = [[waypoints.T[0][1], waypoints.T[1][1]]]
            # plt.plot(waypoints.T[0][1], waypoints.T[1][1], '-or', label="Stationary Point")
            waypoints = self.vehicles[vehicle_color]["snapshots"][0]["center_of_car"]
            plt.plot(waypoints[0], waypoints[1], '-or', label="Stationary Point")
            path = np.array([waypoints[0], waypoints[1]]).reshape(1, -1)

            # plt.grid(True)
            # plt.legend()
            # plt.savefig(self.output_folder + '{}_{}_vehicle_cubic_spline.jpg'.format(self.process_number, vehicle_color), dpi=150)
            # # print(path)
            # # fig1 = plt.gcf()
            # # fig1.savefig('cubic_spline_stationary.jpg' , dpi=150)
            plt.close()
        
        self.process_number += 1
        # _ = plt.set_axis_off()
        # plt.show()
        
        return path
    
    def removeRedundantTrajPoints(self, sample_size):
        for vehicle_color in self.vehicles:
            final_trajectory = self.vehicles[vehicle_color]["trajectories"]["original_trajectory"].tolist()
            print("len of the midpoint_of_lane", len(final_trajectory))
            # last_point = final_trajectory[-1]
            # count = 1
            if (len(final_trajectory) > 1):
                adjusted_trajectory = list()
                for i, point in enumerate(final_trajectory):
                    # print("length of final_trajectory = ", len(final_trajectory))
                    adjusted_points = final_trajectory[:i]
                    ref_point = np.array(final_trajectory[i])
                    # print("ref point", ref_point)
                    traj_points = np.array(final_trajectory[i+1:])                
                    edist = np.sqrt(( (traj_points[:, 0] - ref_point[0]) ** 2 + (traj_points[:, 1] - ref_point[1]) ** 2))
                    edist = edist.reshape(edist.shape[0], -1)
                    traj_points = traj_points.reshape(traj_points.shape[0], -1)
                    edist_traj_points = np.hstack([edist, traj_points])
                    nearest_points = edist_traj_points[edist_traj_points[:, 0].argsort()][:, 0] < sample_size
                    edist_traj_points = edist_traj_points[nearest_points]
                    index_near_points = np.where(nearest_points)[0]
                    furtherest_points = np.delete(traj_points, index_near_points, 0)
                    del final_trajectory
                    final_trajectory = list()
                    if i == 0:
                        final_trajectory.append(ref_point.tolist())
                        final_trajectory.extend(furtherest_points.tolist())
                    else:
                        final_trajectory.extend(adjusted_points)
                        final_trajectory.extend(furtherest_points.tolist())
                    
                    adjusted_trajectory.append(final_trajectory)
                    try:
                        if (final_trajectory[-1] == final_trajectory[i+1]):
                            break
                    except:
                        break
                    #     break
                    # print("ref point", ref_point)
                        # break
                    # break
                # print("final_trajectory = ", adjusted_trajectory)
            
            else:
                adjusted_trajectory = [self.vehicles[vehicle_color]["trajectories"]["original_trajectory"].tolist()]
                    
            self.vehicles[vehicle_color]["trajectories"]["original_trajectory"] = np.array(adjusted_trajectory[-1])

    # sampled_traj = image.copy()

    # for vehicle_color in vehicles:
    #     print("Vehicle's Color", vehicle_color)
    #     sampled_trajectory = np.array(removeRedundantTrajPoints(
    #         vehicles[vehicle_color]["trajectories"]["original_trajectory"].tolist(), car_width / 2))

            # for pt in sampled_trajectory:
            #         ## Visualizing the distorted road
            #     ptx = int(round(pt[0]))
            #     pty = int(round(pt[1]))
            #     cv2.circle(sampled_traj, (ptx, pty), 3, (0, 255, 0), -1)

            # print("points in the sampled trajectory for vehicle ", vehicle_color, " = ",   len(sampled_trajectory))

            # cv2.imshow("distorted and mapped trajectory of vehicle", sampled_traj)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    def trajectoryDistortionMapping(self, image):
        for vehicle_color in self.vehicles:
            trajectory_waypoints = self.vehicles[vehicle_color]["trajectories"]["original_trajectory"]
            for pt in trajectory_waypoints:
                ## Visualizing the distorted road
                ptx = int(round(pt[0]))
                pty = int(round(pt[1]))
                cv2.circle(image, (ptx, pty), 3, (0, 255, 0), -1)
            # self.pre_process.showImage("trajectory of vehicle", image)
            cv2.imwrite(self.output_folder + '{}_original_trajectory.jpg'.format(self.process_number), image)
            
            distortionMapping = lambda x, r: x * r
            aspect_ratio = self.vehicles[vehicle_color]["dimensions"]["car_length_sim"] / self.vehicles[vehicle_color]["dimensions"]["car_length"]
            # aspect_ratio = car_width / car_length
            result = distortionMapping(trajectory_waypoints, aspect_ratio)
            # print("results of distortion and mapping",result)
            for pt in result:
                ## Visualizing the distorted road
                ptx = int(round(pt[0])) 
                pty = int(round(pt[1]))
                cv2.circle(image, (ptx, pty), 2, (0, 255, 0), -1)
            # self.pre_process.showImage("distorted and mapped trajectory of vehicle", image)
            self.vehicles[vehicle_color]["trajectories"]["distorted_trajectory"] = result
            # print("\n Wapoints in the points in the distort trajectory", vehicle_color, " = ",   len(result))
            cv2.imwrite(self.output_folder + '{}_distorted_trajectory.jpg'.format(self.process_number), image)
            self.process_number += 1
            
        if self.show_image:
            self.pre_process.showImage("distorted and mapped trajectory of vehicle", image, time=1000)
        
        # self.pre_process.plotFigure(image, cmap="brg_r", title="distorted and mapped trajectory of vehicle")
        # cv2.imwrite("distort_mapped_trajectory.jpg", image)


    def adjustTrajectoryToSimulation(self):
        """ Adjusting the trajectory for the simulation settings """
        distorted_height = self.height * self.vehicles["red"]["dimensions"]["car_length_sim"] / \
            self.vehicles["red"]["dimensions"]["car_length"]
        for vehicle_color in self.vehicles:
            distorted_trajectory = self.vehicles[vehicle_color]["trajectories"]["distorted_trajectory"]
            predicted_trajectory = np.hstack([distorted_trajectory[:, 0].reshape(distorted_trajectory.shape[0], 1),
                                             (distorted_height - distorted_trajectory[:, 1]).reshape(distorted_trajectory.shape[0], 1)])
            self.vehicles[vehicle_color]["trajectories"]["simulation_trajectory"] = predicted_trajectory


    def calculateTrajectoryArcLength(self):
        """ Calculating the arc length of trajectory"""
        t_arc_0 = time.time()
        for vehicle_color in self.vehicles:
            self.vehicles[vehicle_color]["kinematics"] = dict()
            trajectory  = self.vehicles[vehicle_color]["trajectories"]["simulation_trajectory"]
            snapshots   = self.vehicles[vehicle_color]["snapshots"]
            if (len(snapshots) > 1):
                npts = len(trajectory)  # or  trajectory.shape[0]
                x = trajectory[:, 0]
                y = trajectory[:, 1]
                arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
                for k in range(1, npts):
                    arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
            else:
                ### For stationary point
                arc = 1
            self.vehicles[vehicle_color]["kinematics"]["arc_length"] = arc
            # print("\n\n\n Arc lenght = ", arc)

        t_arc_1 = time.time()
        # print("time take = ", t_arc_1-t_arc_0)  ### The time is always 0.0, empirically tested.
        self.time_efficiency["time_stamp_traj"] = t_arc_1-t_arc_0
        

    def calTimeStampsForTrajectory(self):
        """Kinematics model for calculation of time stamps"""
        timestamp_0 = time.time()
        for vehicle_color in self.vehicles:
            ref_color = ["red", "blue"]
            ref_color.remove(vehicle_color)
            ref_color = ref_color[0]

            primary_arc_length      = self.vehicles[vehicle_color]["kinematics"]["arc_length"]
            ref_arc_length          = self.vehicles[ref_color]["kinematics"]["arc_length"]
            distortory_trajectory   = self.vehicles[vehicle_color]["trajectories"]["distorted_trajectory"]

            arc_lenght_ratio = primary_arc_length / ref_arc_length
            ## speed of 34 ms/s is ideal for internal validity without any delays as the crash sketches are created using the BeamNG simulator itself
            ## whereas for external validity the simulation accuaracy is 83.7 percent for speed of 34 or 32 m/s, to increase the overall simulation accuracy and the 
            ## impact accuracy on the vehicle add delay of the 2.1 meters (i.e. which is the vehicle width) to the vehicle that has the shorter trajectory. This 
            ## due to the fact that vehicle in the external crash sketches dont have an exact length to width ratio, which eventually disturbs the crash impact 
            ## accuracy, IOU and the overall simulation accuracy.  
            speed_mi_hr = 34 # mi/hr #####   30, 32, 34, 36
                
            speed_m_s = speed_mi_hr * (1.6 * 1000) * (1 / (60 * 60))
            
            ### Difference in the arc length of the trajectories divided by speed in m/s  ==> delay of vehicle with smaller trajectory
            diff_arc_len_time       = abs(primary_arc_length - ref_arc_length) / speed_m_s

            if(primary_arc_length > 1):
                # for time travelled by the vehicle within single point
                veh_traj_total_time = primary_arc_length / speed_m_s
                #### for 200 points
                # time_stamp = (veh_traj_total_time + 0) / len(distortory_trajectory)
                
                if primary_arc_length > ref_arc_length:
                    time_stamp = (veh_traj_total_time) / len(distortory_trajectory)
                else:
                    # print("vehicle color = ", vehicle_color, "delay = ", diff_arc_len_time)
                    time_stamp = (veh_traj_total_time + diff_arc_len_time) / len(self.vehicles[ref_color]["trajectories"]["distorted_trajectory"])
                    # time_stamp = (veh_traj_total_time) / len(distortory_trajectory)
                # print(trajectory_arc_length, len(distortory_trajectory), speed_m_s, veh_traj_total_time, time_stamp)
            else:
                primary_arc_length = 1
                distortory_trajectory = [1]
                # speed_m_s = 0
                veh_traj_total_time = 0
                time_stamp = 0

            self.vehicles[vehicle_color]["kinematics"]["speed_m_s"] = speed_m_s
            self.vehicles[vehicle_color]["kinematics"]["veh_traj_total_time"] = veh_traj_total_time
            self.vehicles[vehicle_color]["kinematics"]["time_stamp"] = time_stamp

            print("vehicle ", vehicle_color)
            print((primary_arc_length, len(distortory_trajectory), speed_m_s, veh_traj_total_time, time_stamp))

        timestamp_1 = time.time()
        # print("time take = ", timestamp_1-timestamp_0)  ####  The time is always 0.0, empirically tested.
        self.time_efficiency["time_stamp_traj"] = timestamp_1-timestamp_0


    def convertingToScriptFormat(self):
        """ Converting the trajectory to script format script format"""
        t0 = time.time()
        
        start_time = 0 # 0.4
        for vehicle_color in self.vehicles:
            script = []
            ref_color = ["red", "blue"]
            ref_color.remove(vehicle_color)
            ref_color = ref_color[0]

            distorted_trajectory    = self.vehicles[vehicle_color]["trajectories"]["simulation_trajectory"]
            prim_arc_len            = self.vehicles[vehicle_color]["kinematics"]["arc_length"]
            ref_arc_len             = self.vehicles[ref_color]["kinematics"]["arc_length"]
            prim_veh_total_time     = self.vehicles[vehicle_color]["kinematics"]["veh_traj_total_time"]
            ref_veh_total_time      = self.vehicles[ref_color]["kinematics"]["veh_traj_total_time"]
            time_stamp              = self.vehicles[vehicle_color]["kinematics"]["time_stamp"]

            # delay       = max(prim_veh_total_time, ref_veh_total_time) - min(prim_veh_total_time, ref_veh_total_time)
            delay = abs(min(prim_veh_total_time, ref_veh_total_time) - max(prim_veh_total_time, ref_veh_total_time))
            # delay = 0
            
            print("delay between vehicles   = ", delay)
            # if abs(prim_arc_len - ref_arc_len) < 10:
            #     delay += 0.10
            if max(prim_veh_total_time, ref_veh_total_time) > 1.6:
                # delay += delay * 0.50
                if abs(prim_arc_len - ref_arc_len) < 10:
                    delay += delay * 0.30
                else:
                    # delay = delay * 0.70
                    None
            else:
                if abs(prim_arc_len - ref_arc_len) < 10:
                    delay += delay * 0.20
                else:
                    # delay = delay * 0.70
                    None
            
            # per_arc = min(prim_arc_len , ref_arc_len) / max(prim_arc_len , ref_arc_len)
            # print("% difference between trajectories = ", per_arc)
            # if max(prim_arc_len, ref_arc_len) / min(prim_arc_len, ref_arc_len) > 1.8:
            #     delay = delay * 0.7
            
            # # Difference in the arc length of the trajectories divided by speed in m/s  ==> delay of vehicle with smaller trajectory
            # diff_arc_len_time       = abs(prim_arc_len - ref_arc_len) / self.vehicles[vehicle_color]["kinematics"]["speed_m_s"]
            # if diff_arc_len_time > 1:
            #     diff_arc_len_time = diff_arc_len_time * 0.80
            
            # if abs(prim_arc_len - ref_arc_len) > 10:
            #     if diff_arc_len_time > 2:
            #         diff_arc_len_time = diff_arc_len_time * 0.40
            #     else:
            #         diff_arc_len_time = diff_arc_len_time * 0.70
            
            
            print("Adjusted delay = ", delay)
            # print("delay between last waypoint and crash point  =", delay_last_point)
            # print("prim_arc_len > ref_arc_len = ", prim_arc_len > ref_arc_len)
            
            for i, traj_point in enumerate(distorted_trajectory):
                if prim_arc_len > ref_arc_len:
                    node = {
                        'x': traj_point[0],
                        'y': traj_point[1],
                        'z': 0,
                        "t": i * time_stamp + start_time,
                    }
                else:
                    node = {
                        'x': traj_point[0],
                        'y': traj_point[1],
                        'z': 0,
                        "t": i * (time_stamp) + start_time +  delay # + diff_arc_len_time, #+ delay_last_point,
                    }

                script.append(node)

            self.vehicles[vehicle_color]["trajectories"]["script_trajectory"] = script
            # print("Vehicle first control point = ", script[0])

        t1 = time.time()
        # print("time take = ", t1-t0)
        script_traj = t1-t0
        self.time_efficiency["script_traj"] = script_traj + self.time_efficiency["time_stamp_traj"] + self.time_efficiency["time_stamp_traj"]
        
        
    def computeDebugTrajectories(self):
        for vehicle_color in self.vehicles:
            debug_trajectory = list()
            point_colors = list()
            spheres = list()
            sphere_colors = list()
            self.vehicles["red"]
            for i, point in enumerate(self.vehicles[vehicle_color]["trajectories"]["simulation_trajectory"].tolist()):
                point.extend([0]) 
                # print(point)
                debug_trajectory.append(tuple(point))
                point_colors.append((0, 0, 255, 0.8))
                spheres.append((point[0], point[1], 0, 0.25))
                sphere_colors.append((255, 0, 0, 0.8))

            self.vehicles[vehicle_color]["trajectories"]["debug_trajectory"] = debug_trajectory
            self.vehicles[vehicle_color]["trajectories"]["point_colors"]     = point_colors
            self.vehicles[vehicle_color]["trajectories"]["spheres"]          = spheres
            self.vehicles[vehicle_color]["trajectories"]["sphere_colors"]    = sphere_colors
            # print("Vehicle Trajectory", vehicle_color, ":")
            # print(len(debug_trajectory), "\n") #, len(point_colors), len(spheres), len(sphere_colors))


    def extractKinematicsInformation(self, image_path, vehicles, time_efficiency, output_folder, show_image):
        
        image = self.pre_process.readImage(image_path=image_path)
        self.height, self.width = image.shape[:2]
        self.time_efficiency    = time_efficiency
        self.vehicles           = vehicles
        self.show_image         = show_image
        self.output_folder      = os.path.join(output_folder, "kinematics/")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.process_number = 0
        
        self.selectionVehicleSnapshots()
        self.selectNumberOfWaypoints()
        self.vehicleDistortedControlPoints(image.copy())
        
        #### -------- Bezier Curve Trajectories ------------ #######
        t0 = time.time()
        self.vehicles["red"]["trajectories"]["computed"]["bezier_curve"]    = self.calBezierPath(vehicle_color="red", n_points=90) # 120, 81
        self.vehicles["blue"]["trajectories"]["computed"]["bezier_curve"]   = self.calBezierPath(vehicle_color="blue",  n_points=90) # 120
        t1 = time.time()
        self.time_efficiency["compute_bezier"] = t1-t0
        # print("compute_bezier time = ", t1-t0)
        
        # #### -------- Bezier Spline Trajectories ----------- #######
        # t0 = time.time()
        self.vehicles["red"]["trajectories"]["computed"]["b_spline"] = np.array([0])
        self.vehicles["blue"]["trajectories"]["computed"]["b_spline"] = np.array([0])
        # self.vehicles["red"]["trajectories"]["computed"]["b_spline"]        = np.array(self.calBezierSplineTrajectory(vehicle_color="red", n_course_point=81)) # 81
        # self.vehicles["blue"]["trajectories"]["computed"]["b_spline"]       = np.array(self.calBezierSplineTrajectory(vehicle_color="blue", n_course_point=81))
        # t1 = time.time()
        # self.time_efficiency["compute_b_spline"] = t1-t0
        # # print("compute_bezier time = ", t1-t0)
        
        # #### -------- Cublic Spline Trajectories ----------- #######
        # t0 = time.time()
        self.vehicles["red"]["trajectories"]["computed"]["cubic_spline"]  = np.array([0])
        self.vehicles["blue"]["trajectories"]["computed"]["cubic_spline"] = np.array([0])
        # self.vehicles["red"]["trajectories"]["computed"]["cubic_spline"]    = np.array(self.calCubicSplineTrajectory(vehicle_color="red")) # np.array(cubic_red_veh_traj[::])
        # self.vehicles["blue"]["trajectories"]["computed"]["cubic_spline"]   = np.array(self.calCubicSplineTrajectory(vehicle_color="blue"))
        # t1 = time.time()
        # self.time_efficiency["compute_c_spline"] = t1-t0
        # # print("compute_bezier time = ", t1-t0)
        
        """ Chossing number of trajectory points for the simulation 
            bezier_curve , cubic_spline, b_spline  """

        trajectory_red  = self.vehicles["red"]["trajectories"]["computed"]["bezier_curve"]
        trajectory_blue = self.vehicles["blue"]["trajectories"]["computed"]["bezier_curve"]

        self.vehicles["red"]["trajectories"]["original_trajectory"]  = np.array( [waypoint.tolist() for i, waypoint in enumerate(trajectory_red) if (i % 1 == 0)])
        self.vehicles["blue"]["trajectories"]["original_trajectory"] = np.array( [waypoint.tolist() for i, waypoint in enumerate(trajectory_blue) if (i % 1 == 0)])
        
        self.removeRedundantTrajPoints(sample_size=self.vehicles["red"]["dimensions"]["car_width"] / 2)

        # print(" shape of the red vehicle trajectory is =  ",  self.vehicles["red"]["trajectories"]["original_trajectory"].shape)
        # print(" shape of the blue vehicle trajectory is =  ", self.vehicles["blue"]["trajectories"]["original_trajectory"].shape)
        
        self.trajectoryDistortionMapping(image.copy())
        
        self.adjustTrajectoryToSimulation()
        self.calculateTrajectoryArcLength()
        self.calTimeStampsForTrajectory()
        self.convertingToScriptFormat()
        self.computeDebugTrajectories()
        
        return self.vehicles, self.time_efficiency


        



    

    
