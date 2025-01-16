gnome-terminal -- bash -c "~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh ~/RlAutoDrone_Hemanth/isaacsim_code/camera_feed_with_sim.py; exec bash"
gnome-terminal -- bash -c "cd ~/RlAutoDrone_Hemanth/PX4-AutoPilot && make px4_sitl_default none; exec bash"
gnome-terminal -- bash -c "cd ~/RlAutoDrone_Hemanth && ./QGroundControl.AppImage; exec bash"