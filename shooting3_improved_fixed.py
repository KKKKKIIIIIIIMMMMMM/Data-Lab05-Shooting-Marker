#!/usr/bin/env python3
"""
ระบบควบคุม Gimbal สำหรับยิง Blaster ที่ปรับปรุงแล้ว
- ยิงเป้าหมาย 3 เป้า ตามลำดับ ['left', 'center', 'right', 'center', 'left']
- ใช้ระบบ PID ที่ปรับแต่งได้
- คำนวณมุมเป้าหมายแบบ dynamic
- แสดงกล้องแบบ Real-time
- ปรับปรุงการตรวจจับ Marker
"""

import time
import cv2
import robomaster
from robomaster import robot, vision, gimbal, blaster, camera
import json
import math
import pandas as pd

class AdvancedPIDController:
    """PID Controller ที่ปรับปรุงแล้วพร้อม anti-windup และ derivative kick prevention"""
    
    def __init__(self, kp, ki, kd, setpoint=0.5, output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        # Internal state
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.last_input = 0  # For derivative kick prevention
        
        # Anti-windup
        self.integral_limit = 50
        
    def update(self, current_value):
        """คำนวณ PID output"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return 0
            
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term (derivative on measurement to prevent kick)
        d_input = current_value - self.last_input
        d_term = -self.kd * d_input / dt
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Update state
        self.prev_error = error
        self.last_time = current_time
        self.last_input = current_value
        
        return output
    
    def reset(self):
        """รีเซ็ต PID state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.last_input = 0

# Global variables
g_processed_markers = []
performance_log = []

# กิมบอลแองเกิลปัจจุบัน (อัปเดตจาก sub_angle)
g_yaw_angle = 0.0
g_pitch_angle = 0.0

class MarkerObject:
    """คลาสสำหรับเก็บข้อมูล marker"""
    def __init__(self, x, y, w, h, info):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.info = info
        self.center_x = int(x * 640)
        self.center_y = int(y * 360)

def on_detect_marker(marker_info):
    """Callback function สำหรับตรวจจับ marker"""
    global g_processed_markers
    g_processed_markers.clear()
    for m in marker_info:
        g_processed_markers.append(MarkerObject(*m))

def on_gimbal_angle(angle_info):
    """อัปเดตมุมกิมบอลปัจจุบันจาก encoder: (pitch, yaw, pitch_ground, yaw_ground)"""
    global g_yaw_angle, g_pitch_angle
    try:
        pitch, yaw, _, _ = angle_info
        g_pitch_angle = float(pitch)
        g_yaw_angle = float(yaw)
    except Exception:
        pass

def calculate_target_angles(target_distance, target_spacing):
    """คำนวณมุม yaw สำหรับแต่ละเป้าหมาย"""
    half_spacing = target_spacing / 2
    angle = math.degrees(math.atan(half_spacing / target_distance))
    
    return {
        'left': -angle,
        'center': 0.0,
        'right': angle
    }

def find_target_marker(target_name, markers):
    """หาเป้าหมายที่ต้องการจาก markers ที่ตรวจพบ - ปรับปรุงแล้ว"""
    if not markers:
        return None
    
    # เรียงลำดับ markers ตาม x coordinate (ซ้ายไปขวา)
    sorted_markers = sorted(markers, key=lambda m: m.x)
    
    print(f"🎯 พบ markers จำนวน {len(sorted_markers)} เป้า:")
    for i, m in enumerate(sorted_markers):
        print(f"  Marker {i+1}: x={m.x:.3f}, y={m.y:.3f}")
    
    # เลือกเป้าตามตำแหน่ง
    if target_name == 'left':
        return sorted_markers[0] if len(sorted_markers) >= 1 else None
    elif target_name == 'center':
        if len(sorted_markers) == 1:
            return sorted_markers[0]
        elif len(sorted_markers) == 2:
            return sorted_markers[0]  # เลือกตัวซ้ายเมื่อมี 2 ตัว
        elif len(sorted_markers) >= 3:
            return sorted_markers[1]  # เลือกตัวกลางเมื่อมี 3 ตัว
    elif target_name == 'right':
        if len(sorted_markers) == 2:
            return sorted_markers[1]  # เลือกตัวขวาเมื่อมี 2 ตัว
        elif len(sorted_markers) >= 3:
            return sorted_markers[2]  # เลือกตัวขวาเมื่อมี 3 ตัว
    
    return None

def scan_for_markers(ep_gimbal, duration=5.0):
    """สแกนหา markers เมื่อสูญเสียการติดตาม - ปรับปรุงแล้ว"""
    print("🔍 กำลังสแกนหา markers...")
    scan_patterns = [
        (0, 0),     # Center
        (-20, 0),   # Left more
        (20, 0),    # Right more
        (-30, 0),   # Far left
        (30, 0),    # Far right
        (0, -10),   # Up
        (0, 10),    # Down
        (-15, -5),  # Left up
        (15, -5),   # Right up
        (-15, 5),   # Left down
        (15, 5),    # Right down
    ]
    
    start_time = time.time()
    best_count = 0
    
    while time.time() - start_time < duration:
        for yaw_speed, pitch_speed in scan_patterns:
            ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
            time.sleep(0.5)  # เพิ่มเวลารอให้ gimbal เคลื่อนที่
            
            # ตรวจสอบว่าพบ markers หรือไม่
            markers_1 = [m for m in g_processed_markers if m.info == '1']
            current_count = len(markers_1)
            
            if current_count > best_count:
                best_count = current_count
                print(f"  พบ markers {current_count} เป้า ที่ตำแหน่ง yaw={yaw_speed}, pitch={pitch_speed}")
            
            if current_count >= 3:
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                print(f"✅ พบ markers ครบ {current_count} เป้า!")
                return True
                
            # หยุดสแกนถ้าหมดเวลา
            if time.time() - start_time >= duration:
                break
    
    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
    print(f"⚠️  สแกนเสร็จสิ้น พบ markers สูงสุด {best_count} เป้า")
    return best_count >= 1  # คืนค่า True ถ้าพบอย่างน้อย 1 marker

def log_performance(target_name, success, aim_time, error_x, error_y, trace_file=None):
    """บันทึกประสิทธิภาพการยิง"""
    performance_log.append({
        'target': target_name,
        'success': success,
        'aim_time': aim_time,
        'final_error_x': error_x,
        'final_error_y': error_y,
        'timestamp': time.time(),
        'trace_file': trace_file
    })

def compute_time_response_metrics(trace_rows, value_key, time_key='t', settle_pct=0.02):
    """คำนวณตัวชี้วัด time response: rise time (10-90%), settling time (±2%), overshoot (%)
    - trace_rows: list[dict]
    - value_key: str เช่น 'gimbal_yaw' หรือ 'gimbal_pitch'
    - time_key: str คีย์เวลาสัมพัทธ์ในวินาที
    """
    if not trace_rows or len(trace_rows) < 3:
        return None
    times = [row.get(time_key, None) for row in trace_rows]
    values = [row.get(value_key, None) for row in trace_rows]
    if any(v is None for v in values) or any(t is None for t in times):
        return None
    t0 = times[0]
    v0 = values[0]
    vf = values[-1]
    delta = vf - v0
    if abs(delta) < 1e-6:
        return {
            'rise_time': None,
            'settling_time': None,
            'overshoot_pct': 0.0
        }
    # Rise time: first time reaches 10% and 90% of step magnitude
    v10 = v0 + 0.1 * delta
    v90 = v0 + 0.9 * delta
    t10 = None
    t90 = None
    # Determine monotonic direction
    increasing = delta > 0
    for t, v in zip(times, values):
        if t10 is None:
            if (increasing and v >= v10) or ((not increasing) and v <= v10):
                t10 = t
        if t90 is None:
            if (increasing and v >= v90) or ((not increasing) and v <= v90):
                t90 = t
                break
    rise_time = None
    if t10 is not None and t90 is not None:
        rise_time = max(0.0, t90 - t10)
    # Settling time: first time into ±settle_pct band of vf and stays there
    band = abs(delta) * settle_pct
    settling_time = None
    for idx, (t, v) in enumerate(zip(times, values)):
        if abs(v - vf) <= band:
            # Check stays within band for the rest
            if all(abs(v2 - vf) <= band for v2 in values[idx:]):
                settling_time = max(0.0, t - t0)
                break
    # Overshoot percentage relative to final value (peak deviation beyond vf)
    if increasing:
        peak = max(values)
        overshoot = peak - vf
    else:
        trough = min(values)
        overshoot = vf - trough
    overshoot_pct = max(0.0, (overshoot / abs(delta)) * 100.0)
    return {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot_pct': overshoot_pct
    }

def draw_markers_on_image(img, markers):
    """วาด markers บนภาพ"""
    if img is None:
        return img
        
    for i, marker in enumerate(markers):
        # คำนวณตำแหน่งสำหรับวาดบนภาพ
        x1 = int((marker.x - marker.w/2) * 640)
        y1 = int((marker.y - marker.h/2) * 360)
        x2 = int((marker.x + marker.w/2) * 640)
        y2 = int((marker.y + marker.h/2) * 360)
        center_x = int(marker.x * 640)
        center_y = int(marker.y * 360)
        
        # วาดกรอบ
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # วาดข้อความ
        cv2.putText(img, f"Marker {marker.info}", (center_x-30, center_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"({marker.x:.2f}, {marker.y:.2f})", (center_x-30, center_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # วาดจุดกลาง
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # วาดเส้นกากบาทกลางหน้าจอ
    cv2.line(img, (640-20, 360), (640+20, 360), (255, 0, 0), 2)
    cv2.line(img, (640, 360-20), (640, 360+20), (255, 0, 0), 2)
    
    return img

def main():
    """ฟังก์ชันหลัก"""
    print("=== เริ่มต้นระบบควบคุม Gimbal ===")
    
    # เชื่อมต่อหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_vision = ep_robot.vision
    
    # เริ่มต้นกล้องและ vision
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    ep_gimbal.recenter().wait_for_completed()
    # สมัครรับค่ามุมกิมบอลแบบเรียลไทม์
    ep_gimbal.sub_angle(freq=50, callback=on_gimbal_angle)
    
    # สร้างหน้าต่างแสดงผลกล้อง
    cv2.namedWindow("Robomaster Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robomaster Camera Feed", 640, 360)
    
    print("หุ่นยนต์เริ่มต้นเรียบร้อย")
    time.sleep(2)
    
    # โหลดค่า PID
    try:
        with open("pid_values.json", "r") as f:
            pid_config = json.load(f)
        print("โหลดค่า PID จากไฟล์สำเร็จ")
    except FileNotFoundError:
        print("ไม่พบไฟล์ pid_values.json ใช้ค่าเริ่มต้น")
        pid_config = {
            "yaw": {"kp": 100, "ki": 2, "kd": 60},
            "pitch": {"kp": 100, "ki": 2, "kd": 60}
        }
    
    # คำนวณมุมเป้าหมาย
    TARGET_DISTANCE = 1.0  # เมตร
    TARGET_SPACING = 0.6   # เมตร
    target_angles = calculate_target_angles(TARGET_DISTANCE, TARGET_SPACING)
    
    print(f"มุมเป้าหมาย: ซ้าย={target_angles['left']:.1f}°, กลาง={target_angles['center']:.1f}°, ขวา={target_angles['right']:.1f}°")
    
    # สร้าง PID controllers
    pid_yaw = AdvancedPIDController(
        kp=pid_config["yaw"]["kp"],
        ki=pid_config["yaw"]["ki"], 
        kd=pid_config["yaw"]["kd"]
    )
    
    pid_pitch = AdvancedPIDController(
        kp=pid_config["pitch"]["kp"],
        ki=pid_config["pitch"]["ki"],
        kd=pid_config["pitch"]["kd"]
    )
    
    # ลำดับการยิง
    shooting_sequence = ['left', 'center', 'right', 'center', 'left']
    
    # ตั้งค่าการเล็ง
    AIM_TOLERANCE = 0.03  # ความแม่นยำในการเล็ง
    MAX_AIM_TIME = 15.0    # เวลาสูงสุดในการเล็ง (วินาที)
    
    print(f"\n=== เริ่มลำดับการยิง ===")
    print(f"ลำดับ: {shooting_sequence}")
    
    # สแกนหา markers เริ่มต้น
    print("\n🔍 สแกนหา markers เริ่มต้น...")
    scan_for_markers(ep_gimbal, duration=8.0)

    # เตรียมตัวเก็บ trace รวมทุกช็อต และชื่อไฟล์รวม
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    all_trace_rows = []
    
    for shot_num, target_name in enumerate(shooting_sequence, 1):
        print(f"\n--- การยิงครั้งที่ {shot_num}: เป้า {target_name.upper()} ---")
        
        # รีเซ็ต PID controllers
        pid_yaw.reset()
        pid_pitch.reset()
        
        # เคลื่อนไปยังตำแหน่งเป้าหมายโดยประมาณ
        target_yaw = target_angles[target_name]
        print(f"เคลื่อนไปยังมุม {target_yaw:.1f}°")
        ep_gimbal.moveto(pitch=-10, yaw=target_yaw).wait_for_completed()
        time.sleep(0.5)
        
        # ตรวจสอบ markers
        markers_1 = [m for m in g_processed_markers if m.info == '1']
        print(f"ตรวจพบ markers จำนวน {len(markers_1)} เป้า")
        
        if len(markers_1) == 0:
            print("ไม่พบ markers เลย กำลังสแกน...")
            if not scan_for_markers(ep_gimbal):
                print("ไม่สามารถหา markers ได้ ข้ามเป้านี้")
                continue
        
        # เริ่มการเล็งและยิง
        aim_start_time = time.time()
        locked = False
        trace_rows = []  # เก็บ time response ต่อช็อต
        
        for attempt in range(500):  # จำกัดจำนวนครั้งในการลอง
            # อ่านภาพจากกล้องและแสดงผล
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.1)
            if img is not None:
                # วาด markers บนภาพ
                current_markers = [m for m in g_processed_markers if m.info == '1']
                img_with_markers = draw_markers_on_image(img, current_markers)
                
                # แสดงข้อมูลเป้าหมายปัจจุบัน
                cv2.putText(img_with_markers, f"Target: {target_name.upper()}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img_with_markers, f"Shot: {shot_num}/5", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img_with_markers, f"Markers: {len(current_markers)}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Robomaster Camera Feed", img_with_markers)
                cv2.waitKey(1)
            
            # หาเป้าหมายปัจจุบัน
            current_markers = [m for m in g_processed_markers if m.info == '1']
            target_marker = find_target_marker(target_name, current_markers)
            
            if not target_marker:
                print(f"❌ ไม่พบเป้า {target_name} กำลังสแกนใหม่...")
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                if not scan_for_markers(ep_gimbal, duration=2.0):
                    break
                continue
            
            # คำนวณ error
            error_x = target_marker.x - 0.5
            error_y = target_marker.y - 0.5
            
            # คำนวณ PID output
            yaw_speed = -pid_yaw.update(target_marker.x)
            pitch_speed = pid_pitch.update(target_marker.y)
            
            # ส่งคำสั่งไปยัง gimbal
            ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)

            # เก็บ time response หนึ่งจุด
            trace_rows.append({
                't': time.time() - aim_start_time,
                'shot': int(shot_num),
                'target': str(target_name),
                'marker_x': float(target_marker.x),
                'marker_y': float(target_marker.y),
                'error_x': float(error_x),
                'error_y': float(error_y),
                'pid_yaw_out': float(yaw_speed),
                'pid_pitch_out': float(pitch_speed),
                'gimbal_yaw': float(g_yaw_angle),
                'gimbal_pitch': float(g_pitch_angle),
                'kp_yaw': float(pid_config["yaw"]["kp"]),
                'ki_yaw': float(pid_config["yaw"]["ki"]),
                'kd_yaw': float(pid_config["yaw"]["kd"]),
                'kp_pitch': float(pid_config["pitch"]["kp"]),
                'ki_pitch': float(pid_config["pitch"]["ki"]),
                'kd_pitch': float(pid_config["pitch"]["kd"])
            })
            
            print(f"เล็ง {target_name}: Error X={error_x:.3f}, Y={error_y:.3f}, Yaw Speed={yaw_speed:.1f}, Pitch Speed={pitch_speed:.1f}")
            
            # ตรวจสอบว่าเล็งได้แม่นยำพอหรือไม่
            if abs(error_x) <= AIM_TOLERANCE and abs(error_y) <= AIM_TOLERANCE:
                print(f"🎯 ล็อคเป้า {target_name.upper()} สำเร็จ!")
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                time.sleep(0.2)  # รอให้ gimbal หยุดสั่น
                
                # ยิง!
                ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                print("🔥 ยิง!")
                
                # บันทึกประสิทธิภาพและ time response
                aim_time = time.time() - aim_start_time
                trace_filename = f"time_response_shot_{shot_num}_{target_name}.csv"
                try:
                    pd.DataFrame(trace_rows).to_csv(trace_filename, index=False)
                except Exception:
                    trace_filename = None
                # คำนวณ metrics ของมุมกิมบอล
                yaw_metrics = compute_time_response_metrics(trace_rows, 'gimbal_yaw') or {}
                pitch_metrics = compute_time_response_metrics(trace_rows, 'gimbal_pitch') or {}
                log_entry_extra = {
                    'yaw_rise_time': yaw_metrics.get('rise_time'),
                    'yaw_settling_time': yaw_metrics.get('settling_time'),
                    'yaw_overshoot_pct': yaw_metrics.get('overshoot_pct'),
                    'pitch_rise_time': pitch_metrics.get('rise_time'),
                    'pitch_settling_time': pitch_metrics.get('settling_time'),
                    'pitch_overshoot_pct': pitch_metrics.get('overshoot_pct')
                }
                log_performance(target_name, True, aim_time, error_x, error_y, trace_filename)
                # รวม metrics เข้า entry ล่าสุด
                performance_log[-1].update(log_entry_extra)
                
                locked = True
                time.sleep(1.0)  # หยุดสั้นๆ หลังยิง
                break
            
            # ตรวจสอบ timeout
            if time.time() - aim_start_time > MAX_AIM_TIME:
                print(f"⏰ หมดเวลาการเล็งเป้า {target_name.upper()}")
                break
                
            time.sleep(0.01)  # ลด CPU usage
        
        # รวม trace ของช็อตนี้เข้า all_trace_rows
        all_trace_rows.extend(trace_rows)

        if not locked:
            aim_time = time.time() - aim_start_time
            trace_filename = f"time_response_shot_{shot_num}_{target_name}.csv"
            try:
                pd.DataFrame(trace_rows).to_csv(trace_filename, index=False)
            except Exception:
                trace_filename = None
            yaw_metrics = compute_time_response_metrics(trace_rows, 'gimbal_yaw') or {}
            pitch_metrics = compute_time_response_metrics(trace_rows, 'gimbal_pitch') or {}
            log_entry_extra = {
                'yaw_rise_time': yaw_metrics.get('rise_time'),
                'yaw_settling_time': yaw_metrics.get('settling_time'),
                'yaw_overshoot_pct': yaw_metrics.get('overshoot_pct'),
                'pitch_rise_time': pitch_metrics.get('rise_time'),
                'pitch_settling_time': pitch_metrics.get('settling_time'),
                'pitch_overshoot_pct': pitch_metrics.get('overshoot_pct')
            }
            log_performance(target_name, False, aim_time, 999, 999, trace_filename)
            performance_log[-1].update(log_entry_extra)
            print(f"❌ ไม่สามารถล็อคเป้า {target_name.upper()} ได้")
        
        ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
    
    # สรุปผลการยิง
    print("\n=== สรุปผลการยิง ===")
    successful_shots = sum(1 for log in performance_log if log['success'])
    total_shots = len(performance_log)
    
    print(f"ยิงสำเร็จ: {successful_shots}/{total_shots} ครั้ง")
    if total_shots > 0:
        print(f"อัตราความสำเร็จ: {successful_shots/total_shots*100:.1f}%")
    
    # บันทึกผลลัพธ์
    if performance_log:
        df = pd.DataFrame(performance_log)
        df.to_csv("shooting_performance.csv", index=False)
        print("บันทึกผลลัพธ์ใน shooting_performance.csv")

    # บันทึก time response รวมทุกช็อต
    if all_trace_rows:
        combined_csv = f"gimbal_time_response_{session_ts}.csv"
        try:
            pd.DataFrame(all_trace_rows).to_csv(combined_csv, index=False)
            print(f"บันทึก time response รวมทุกช็อตใน {combined_csv}")
        except Exception as e:
            print(f"บันทึกไฟล์ time response รวมล้มเหลว: {e}")
    
    # ปิดระบบ
    cv2.destroyAllWindows()
    ep_vision.unsub_detect_info(name="marker")
    ep_gimbal.unsub_angle()
    ep_camera.stop_video_stream()
    ep_robot.close()
    
    print("\n🏁 ภารกิจเสร็จสิ้น")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  หยุดการทำงานโดยผู้ใช้")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

