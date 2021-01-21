#!/usr/bin/env python
from record import Record
from wav2csv import wav2csv
import rospy

def main():
    Record()
    finish = rosparam.get("~FINISH")
    if finish == True:
        wav2csv()

if __name__ == "__main__":
    main()
