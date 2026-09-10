#!/usr/bin/env python3
"""
Format filenames according to the pattern established.
"""
from glob import glob
from tqdm import tqdm
import argparse
import shutil
import os



odnu = {

"riksdagen-motions-pdf/data/200304/A1.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_1.pdf",
"riksdagen-motions-pdf/data/200304/A2.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_2.pdf",
"riksdagen-motions-pdf/data/200304/A205.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A20_5.pdf",
"riksdagen-motions-pdf/data/200304/A206.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A20_6.pdf",
"riksdagen-motions-pdf/data/200304/A207.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A20_7.pdf",
"riksdagen-motions-pdf/data/200304/A208.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A20_8.pdf",
"riksdagen-motions-pdf/data/200304/A209.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A20_9.pdf",
"riksdagen-motions-pdf/data/200304/A210.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_0.pdf",
"riksdagen-motions-pdf/data/200304/A211.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_1.pdf",
"riksdagen-motions-pdf/data/200304/A212.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_2.pdf",
"riksdagen-motions-pdf/data/200304/A213.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_3.pdf",
"riksdagen-motions-pdf/data/200304/A214.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_4.pdf",
"riksdagen-motions-pdf/data/200304/A215.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_5.pdf",
"riksdagen-motions-pdf/data/200304/A216.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_6.pdf",
"riksdagen-motions-pdf/data/200304/A217.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_7.pdf",
"riksdagen-motions-pdf/data/200304/A218.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_8.pdf",
"riksdagen-motions-pdf/data/200304/A219.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A21_9.pdf",
"riksdagen-motions-pdf/data/200304/A220.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_0.pdf",
"riksdagen-motions-pdf/data/200304/A221.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_1.pdf",
"riksdagen-motions-pdf/data/200304/A222.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_2.pdf",
"riksdagen-motions-pdf/data/200304/A223.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_3.pdf",
"riksdagen-motions-pdf/data/200304/A224.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_4.pdf",
"riksdagen-motions-pdf/data/200304/A225.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_5.pdf",
"riksdagen-motions-pdf/data/200304/A226.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_6.pdf",
"riksdagen-motions-pdf/data/200304/A227.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_7.pdf",
"riksdagen-motions-pdf/data/200304/A228.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_8.pdf",
"riksdagen-motions-pdf/data/200304/A229.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A22_9.pdf",
"riksdagen-motions-pdf/data/200304/A230.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_0.pdf",
"riksdagen-motions-pdf/data/200304/A231.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_1.pdf",
"riksdagen-motions-pdf/data/200304/A232.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_2.pdf",
"riksdagen-motions-pdf/data/200304/A233.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_3.pdf",
"riksdagen-motions-pdf/data/200304/A234.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_4.pdf",
"riksdagen-motions-pdf/data/200304/A235.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_5.pdf",
"riksdagen-motions-pdf/data/200304/A236.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_6.pdf",
"riksdagen-motions-pdf/data/200304/A237.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_7.pdf",
"riksdagen-motions-pdf/data/200304/A238.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_8.pdf",
"riksdagen-motions-pdf/data/200304/A239.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A23_9.pdf",
"riksdagen-motions-pdf/data/200304/A240.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_0.pdf",
"riksdagen-motions-pdf/data/200304/A241.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_1.pdf",
"riksdagen-motions-pdf/data/200304/A242.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_2.pdf",
"riksdagen-motions-pdf/data/200304/A243.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_3.pdf",
"riksdagen-motions-pdf/data/200304/A244.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_4.pdf",
"riksdagen-motions-pdf/data/200304/A245.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_5.pdf",
"riksdagen-motions-pdf/data/200304/A246.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_6.pdf",
"riksdagen-motions-pdf/data/200304/A248.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_8.pdf",
"riksdagen-motions-pdf/data/200304/A249.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A24_9.pdf",
"riksdagen-motions-pdf/data/200304/A250.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_0.pdf",
"riksdagen-motions-pdf/data/200304/A251.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_1.pdf",
"riksdagen-motions-pdf/data/200304/A252.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_2.pdf",
"riksdagen-motions-pdf/data/200304/A253.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_3.pdf",
"riksdagen-motions-pdf/data/200304/A254.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_4.pdf",
"riksdagen-motions-pdf/data/200304/A255.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_5.pdf",
"riksdagen-motions-pdf/data/200304/A256.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_6.pdf",
"riksdagen-motions-pdf/data/200304/A257.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_7.pdf",
"riksdagen-motions-pdf/data/200304/A258.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_8.pdf",
"riksdagen-motions-pdf/data/200304/A259.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A25_9.pdf",
"riksdagen-motions-pdf/data/200304/A260.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_0.pdf",
"riksdagen-motions-pdf/data/200304/A261.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_1.pdf",
"riksdagen-motions-pdf/data/200304/A262.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_2.pdf",
"riksdagen-motions-pdf/data/200304/A263.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_3.pdf",
"riksdagen-motions-pdf/data/200304/A264.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_4.pdf",
"riksdagen-motions-pdf/data/200304/A265.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_5.pdf",
"riksdagen-motions-pdf/data/200304/A266.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_6.pdf",
"riksdagen-motions-pdf/data/200304/A267.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_7.pdf",
"riksdagen-motions-pdf/data/200304/A268.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_8.pdf",
"riksdagen-motions-pdf/data/200304/A269.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A26_9.pdf",
"riksdagen-motions-pdf/data/200304/A270.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_0.pdf",
"riksdagen-motions-pdf/data/200304/A271.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_1.pdf",
"riksdagen-motions-pdf/data/200304/A272.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_2.pdf",
"riksdagen-motions-pdf/data/200304/A273.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_3.pdf",
"riksdagen-motions-pdf/data/200304/A274.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_4.pdf",
"riksdagen-motions-pdf/data/200304/A275.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_5.pdf",
"riksdagen-motions-pdf/data/200304/A276.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_6.pdf",
"riksdagen-motions-pdf/data/200304/A277.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_7.pdf",
"riksdagen-motions-pdf/data/200304/A278.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_8.pdf",
"riksdagen-motions-pdf/data/200304/A279.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A27_9.pdf",
"riksdagen-motions-pdf/data/200304/A280.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_0.pdf",
"riksdagen-motions-pdf/data/200304/A281.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_1.pdf",
"riksdagen-motions-pdf/data/200304/A282.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_2.pdf",
"riksdagen-motions-pdf/data/200304/A283.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_3.pdf",
"riksdagen-motions-pdf/data/200304/A284.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_4.pdf",
"riksdagen-motions-pdf/data/200304/A285.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_5.pdf",
"riksdagen-motions-pdf/data/200304/A286.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_6.pdf",
"riksdagen-motions-pdf/data/200304/A287.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_7.pdf",
"riksdagen-motions-pdf/data/200304/A288.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_8.pdf",
"riksdagen-motions-pdf/data/200304/A289.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A28_9.pdf",
"riksdagen-motions-pdf/data/200304/A290.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_0.pdf",
"riksdagen-motions-pdf/data/200304/A291.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_1.pdf",
"riksdagen-motions-pdf/data/200304/A292.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_2.pdf",
"riksdagen-motions-pdf/data/200304/A293.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_3.pdf",
"riksdagen-motions-pdf/data/200304/A294.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_4.pdf",
"riksdagen-motions-pdf/data/200304/A295.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_5.pdf",
"riksdagen-motions-pdf/data/200304/A296.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_6.pdf",
"riksdagen-motions-pdf/data/200304/A297.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_7.pdf",
"riksdagen-motions-pdf/data/200304/A298.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_8.pdf",
"riksdagen-motions-pdf/data/200304/A299.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A29_9.pdf",
"riksdagen-motions-pdf/data/200304/A3.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_3.pdf",
"riksdagen-motions-pdf/data/200304/A300.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_0.pdf",
"riksdagen-motions-pdf/data/200304/A301.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_1.pdf",
"riksdagen-motions-pdf/data/200304/A302.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_2.pdf",
"riksdagen-motions-pdf/data/200304/A303.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_3.pdf",
"riksdagen-motions-pdf/data/200304/A304.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_4.pdf",
"riksdagen-motions-pdf/data/200304/A305.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_5.pdf",
"riksdagen-motions-pdf/data/200304/A306.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_6.pdf",
"riksdagen-motions-pdf/data/200304/A307.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_7.pdf",
"riksdagen-motions-pdf/data/200304/A308.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_8.pdf",
"riksdagen-motions-pdf/data/200304/A309.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A30_9.pdf",
"riksdagen-motions-pdf/data/200304/A310.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_0.pdf",
"riksdagen-motions-pdf/data/200304/A311.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_1.pdf",
"riksdagen-motions-pdf/data/200304/A312.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_2.pdf",
"riksdagen-motions-pdf/data/200304/A313.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_3.pdf",
"riksdagen-motions-pdf/data/200304/A314.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_4.pdf",
"riksdagen-motions-pdf/data/200304/A315.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_5.pdf",
"riksdagen-motions-pdf/data/200304/A316.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_6.pdf",
"riksdagen-motions-pdf/data/200304/A317.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_7.pdf",
"riksdagen-motions-pdf/data/200304/A318.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_8.pdf",
"riksdagen-motions-pdf/data/200304/A319.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A31_9.pdf",
"riksdagen-motions-pdf/data/200304/A320.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_0.pdf",
"riksdagen-motions-pdf/data/200304/A321.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_1.pdf",
"riksdagen-motions-pdf/data/200304/A322.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_2.pdf",
"riksdagen-motions-pdf/data/200304/A323.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_3.pdf",
"riksdagen-motions-pdf/data/200304/A324.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_4.pdf",
"riksdagen-motions-pdf/data/200304/A325.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_5.pdf",
"riksdagen-motions-pdf/data/200304/A326.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_6.pdf",
"riksdagen-motions-pdf/data/200304/A327.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_7.pdf",
"riksdagen-motions-pdf/data/200304/A328.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_8.pdf",
"riksdagen-motions-pdf/data/200304/A329.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A32_9.pdf",
"riksdagen-motions-pdf/data/200304/A330.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_0.pdf",
"riksdagen-motions-pdf/data/200304/A331.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_1.pdf",
"riksdagen-motions-pdf/data/200304/A332.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_2.pdf",
"riksdagen-motions-pdf/data/200304/A333.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_3.pdf",
"riksdagen-motions-pdf/data/200304/A334.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_4.pdf",
"riksdagen-motions-pdf/data/200304/A335.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_5.pdf",
"riksdagen-motions-pdf/data/200304/A336.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_6.pdf",
"riksdagen-motions-pdf/data/200304/A337.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_7.pdf",
"riksdagen-motions-pdf/data/200304/A338.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_8.pdf",
"riksdagen-motions-pdf/data/200304/A339.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A33_9.pdf",
"riksdagen-motions-pdf/data/200304/A340.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_0.pdf",
"riksdagen-motions-pdf/data/200304/A341.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_1.pdf",
"riksdagen-motions-pdf/data/200304/A342.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_2.pdf",
"riksdagen-motions-pdf/data/200304/A343.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_3.pdf",
"riksdagen-motions-pdf/data/200304/A344.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_4.pdf",
"riksdagen-motions-pdf/data/200304/A345.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_5.pdf",
"riksdagen-motions-pdf/data/200304/A346.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_6.pdf",
"riksdagen-motions-pdf/data/200304/A347.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_7.pdf",
"riksdagen-motions-pdf/data/200304/A348.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_8.pdf",
"riksdagen-motions-pdf/data/200304/A349.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A34_9.pdf",
"riksdagen-motions-pdf/data/200304/A350.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_0.pdf",
"riksdagen-motions-pdf/data/200304/A351.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_1.pdf",
"riksdagen-motions-pdf/data/200304/A352.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_2.pdf",
"riksdagen-motions-pdf/data/200304/A353.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_3.pdf",
"riksdagen-motions-pdf/data/200304/A354.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_4.pdf",
"riksdagen-motions-pdf/data/200304/A355.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_5.pdf",
"riksdagen-motions-pdf/data/200304/A356.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_6.pdf",
"riksdagen-motions-pdf/data/200304/A357.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_7.pdf",
"riksdagen-motions-pdf/data/200304/A358.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_8.pdf",
"riksdagen-motions-pdf/data/200304/A359.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A35_9.pdf",
"riksdagen-motions-pdf/data/200304/A360.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_0.pdf",
"riksdagen-motions-pdf/data/200304/A361.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_1.pdf",
"riksdagen-motions-pdf/data/200304/A362.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_2.pdf",
"riksdagen-motions-pdf/data/200304/A363.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_3.pdf",
"riksdagen-motions-pdf/data/200304/A364.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_4.pdf",
"riksdagen-motions-pdf/data/200304/A365.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_5.pdf",
"riksdagen-motions-pdf/data/200304/A366.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_6.pdf",
"riksdagen-motions-pdf/data/200304/A367.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_7.pdf",
"riksdagen-motions-pdf/data/200304/A368.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_8.pdf",
"riksdagen-motions-pdf/data/200304/A369.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A36_9.pdf",
"riksdagen-motions-pdf/data/200304/A370.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A37_0.pdf",
"riksdagen-motions-pdf/data/200304/A371.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A37_1.pdf",
"riksdagen-motions-pdf/data/200304/A4.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_4.pdf",
"riksdagen-motions-pdf/data/200304/A5.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_5.pdf",
"riksdagen-motions-pdf/data/200304/A6.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_6.pdf",
"riksdagen-motions-pdf/data/200304/A7.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_A_7.pdf",
"riksdagen-motions-pdf/data/200304/BO1.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_BO_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo10.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo1_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo201.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo202.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo203.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo204.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo205.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo206.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo207.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo208.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo209.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo20_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo210.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo211.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo212.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo213.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo214.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo215.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo216.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo217.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo218.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo219.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo21_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo220.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo221.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo222.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo223.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo224.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo225.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo226.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo227.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo228.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo229.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo22_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo230.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo231.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo232.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo233.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo234.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo235.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo236.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo237.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo238.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo239.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo23_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo240.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo241.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo242.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo243.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo244.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo245.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo246.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo247.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo248.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo249.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo24_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo250.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo251.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo252.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo253.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo254.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo255.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo256.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo257.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_7.pdf",
"riksdagen-motions-pdf/data/200304/Bo258.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_8.pdf",
"riksdagen-motions-pdf/data/200304/Bo259.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo25_9.pdf",
"riksdagen-motions-pdf/data/200304/Bo260.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_0.pdf",
"riksdagen-motions-pdf/data/200304/Bo261.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_1.pdf",
"riksdagen-motions-pdf/data/200304/Bo262.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_2.pdf",
"riksdagen-motions-pdf/data/200304/Bo263.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_3.pdf",
"riksdagen-motions-pdf/data/200304/Bo264.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_4.pdf",
"riksdagen-motions-pdf/data/200304/Bo265.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_5.pdf",
"riksdagen-motions-pdf/data/200304/Bo266.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_6.pdf",
"riksdagen-motions-pdf/data/200304/Bo267.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_Bo26_7.pdf",
"riksdagen-motions-pdf/data/200304/a201.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_a20_1.pdf",
"riksdagen-motions-pdf/data/200304/a202.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_a20_2.pdf",
"riksdagen-motions-pdf/data/200304/a203.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_a20_3.pdf",
"riksdagen-motions-pdf/data/200304/a204.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_a20_4.pdf",
"riksdagen-motions-pdf/data/200304/a247.pdf": "riksdagen-motions-pdf/data/200304/mot_200304_a24_7.pdf"

}

undo = {}

for k, v in odnu.items():
    undo[v] = k

def main(args):
    pdf_path = f"{args.pdf_path}/{args.parliament_year}"
    pdf_files = sorted(glob(f"{pdf_path}/*.pdf"))
    for f in tqdm(pdf_files):
        if f in undo:
            #shutil.move(f, undo[f])
            pass
        else:
            base_f = os.path.basename(f)
            n, e = base_f.split('.')
            if base_f.startswith("MOT_"):
                mot, py, committee, index = n.split("_")
                new_f = base_f.lower()
                print(f, f"{pdf_path}/mot_{args.parliament_year}_{committee}_{index:0>4}.{e}")
                shutil.move(f, f"{pdf_path}/mot_{args.parliament_year}_{committee}_{index:0>4}.{e}")
            elif not base_f.startswith("mot"):
                split_point = None

                for i, _ in enumerate(n):
                    try:
                        assert split_point is None
                        _ = int(_)
                        split_point = i
                    except:
                        pass
                committee = n[:split_point]
                index = n[split_point:]
                print(f, f"{pdf_path}/mot_{args.parliament_year}_{committee}_{index:0>4}.{e}")
                shutil.move(f, f"{pdf_path}/mot_{args.parliament_year}_{committee}_{index:0>4}.{e}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("-p", "--parliament-year", required=True)
    parser.add_argument("--pdf-path", default="riksdagen-motions-pdf/data")
    args = parser.parse_args()
    main(args)
