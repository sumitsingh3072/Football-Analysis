from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner    # Import the Tracker and TeamAssigner classes
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import Camera_movement_estimator
from view_transformer import ViewtTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    video_path = r"input_videos\08fd33_4.mp4"
    frames = read_video(video_path)

    # Intialize the tracker
    tracker = Tracker(r"models\best.pt")  

    tracks = tracker.get_object_tracks(frames , 
                                       read_from_stub=True ,
                                        stub_path = 'stubs/tracks_stubs.pkl')
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    
    # Camera Movement Estimation
    camera_movement_estimator = Camera_movement_estimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement.pkl')
    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewtTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)


    #Interpolate Ball Position
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Speed and Distance Estimation
    speed_and_distance_Estimator = SpeedAndDistance_Estimator()
    speed_and_distance_Estimator.add_speed_and_distance_to_tracks(tracks)



    # Assign  Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball Acquistion

    player_assigner = PlayerBallAssigner()

    team_ball_control = []

    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    #Draw output on frames
    output_video_frames  = tracker.draw_annotations(frames,tracks,team_ball_control)

    # Draw camera movement 
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    # Draw Speed and Distance
    speed_and_distance_Estimator.draw_speed_and_distance(output_video_frames,tracks)

    save_video(output_video_frames, 'output_videos/output_video.avi')  

if __name__ == '__main__':
    main()
