"""90-minute countdown. Run in a second terminal: python3 timer.py"""
import sys
import time

TOTAL = 90 * 60
end = time.time() + TOTAL

while True:
    left = int(round(end - time.time()))
    if left <= 0:
        sys.stdout.write("\rTIME UP -- stop typing.            \n")
        break
    mark = ""
    if left in (60 * 60, 45 * 60, 30 * 60, 15 * 60, 5 * 60):
        mark = "  <-- checkpoint"
    sys.stdout.write("\r{:02d}:{:02d} remaining{}".format(left // 60, left % 60, mark))
    sys.stdout.flush()
    time.sleep(1)
