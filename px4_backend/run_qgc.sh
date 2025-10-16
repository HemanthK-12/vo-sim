#!/bin/bash
# Temporarily unset Qt plugin paths
export QT_QPA_PLATFORM_PLUGIN_PATH=""
export QT_PLUGIN_PATH=""
# Run QGroundControl
~/RlAutoDrone_Hemanth/QGroundControl.AppImage
