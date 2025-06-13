import unittest
from unittest.mock import patch, MagicMock

from python import motion as robot_control
import config
import hardware

class TestRobotControl(unittest.TestCase):
    @patch('hardware.aux_motor')
    @patch('hardware.tank')
    @patch('hardware.get_heading')
    def test_drive_distance_zero(self, mock_heading, mock_tank, mock_aux):
        # Setup get_heading and motor positions
        mock_heading.return_value = 0.0
        # Simulate left and right motor positions equal (no movement)
        left = MagicMock(position=0)
        right = MagicMock(position=0)
        mock_tank.left_motor = left
        mock_tank.right_motor = right
        # Call drive_distance with zero distance
        robot_control.drive_distance(0)
        # aux on and off called, but tank.on never called
        mock_aux.on.assert_called()
        mock_aux.off.assert_called()
        mock_tank.on.assert_not_called()
        mock_tank.off.assert_called()

    @patch('hardware.aux_motor')
    @patch('hardware.tank')
    @patch('hardware.get_heading')
    def test_perform_turn_tolerance(self, mock_heading, mock_tank, mock_aux):
        # Simulate already within tolerance
        config.ANGLE_TOLERANCE = 5.0
        # get_heading returns target
        mock_heading.side_effect = [10.0]
        # Should start and immediately stop
        robot_control.perform_turn(0)
        mock_aux.on.assert_called()
        mock_tank.on.assert_not_called()
        mock_tank.off.assert_called()
        mock_aux.off.assert_called()

    @patch('robot_control.drive_distance')
    @patch('robot_control.perform_turn')
    @patch('hardware.calibrate_gyro')
    @patch('hardware.get_heading')
    def test_follow_path_simple(self, mock_heading, mock_cal, mock_turn, mock_drive):
        # Path of two points one cell apart
        config.CELL_SIZE_CM = 10.0
        config.ANGLE_TOLERANCE = 1.0
        # Heading from get_heading
        mock_heading.return_value = 0.0
        points = [(0, 0), (1, 0)]
        robot_control.follow_path(points, start_heading_deg=0)
        # Should calibrate gyro, turn not called (aligned), drive called once
        mock_cal.assert_called()
        mock_turn.assert_not_called()
        mock_drive.assert_called_once_with(10.0, speed_pct=config.DRIVE_SPEED_PCT, target_angle=0.0)

    def test_handle_command_buffering(self):
        buf = {'distance_buffer': 0}
        # Distance only
        robot_control.handle_command({'distance': '5'}, buf)
        self.assertEqual(buf['distance_buffer'], 5.0)
        # Turn flushes buffer
        with patch('robot_control.perform_turn') as mock_turn, patch('robot_control.drive_distance') as mock_drive:
            robot_control.handle_command({'turn': '90'}, buf)
            mock_drive.assert_called_with(5.0)
            mock_turn.assert_called_with(90.0)
            self.assertEqual(buf['distance_buffer'], 0)

    @patch('hardware.aux_motor')
    def test_reverse_aux(self, mock_aux):
        # Test that reverse runs forward then off
        config.AUX_REVERSE_SEC = 0.01
        config.AUX_REVERSE_PCT = 50
        robot_control._reverse_aux()
        mock_aux.on.assert_called_with(50)
        mock_aux.off.assert_called()

if __name__ == '__main__':
    unittest.main()
