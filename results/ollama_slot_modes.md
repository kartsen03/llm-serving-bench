Ollama, per run: each slot id with one letter per request it served at that concurrency, S = slow ITL mode, f = fast (threshold as in the ITL-modes table).

- launch 1, repeat 1, c=8: 0:ffff 1:ffff 2:ffff 3:ffff 4:ffff 5:ffff 6:ffff 7:ffff 8:ffff 9:ffff 10:ffff 11:ffff 12:ffff 13:ffff 14:ffff 15:ffff
- launch 1, repeat 1, c=4: 0:ff 1:ff 2:ff 3:ff 4:ff 5:ff 6:ff 7:ff 8:ff 9:ff 10:ff 11:ff 12:ff 13:ff 14:ff 15:ff
- launch 1, repeat 1, c=2: 0:f 1:f 2:f 3:f 4:f 5:f 6:f 7:f 8:f 9:f 10:f 11:f 12:f 13:f 14:f 15:f
- launch 1, repeat 2, c=4: 0:SS 1:ff 2:ff 3:ff 4:ff 5:ff 6:ff 7:ff 8:ff 9:ff 10:ff 11:ff 12:ff 13:SS 14:SS 15:SS
- launch 1, repeat 2, c=2: 0:S 1:f 2:f 3:f 4:f 5:f 6:f 7:f 8:f 9:f 10:f 11:f 12:f 13:f 14:f 15:S
- launch 1, repeat 2, c=8: 0:SSSS 1:SSSS 2:SSSS 3:ffff 4:ffff 5:ffff 6:ffff 7:ffff 8:ffff 9:ffff 10:ffff 11:SSSS 12:SSSS 13:SSSS 14:SSSS 15:SSSS
- launch 3, repeat 3, c=2: 0:f 1:f 2:f 3:f 4:f 5:f 6:f 7:f 8:f 9:f 10:f 11:f 12:f 13:f 14:f 15:f
- launch 3, repeat 3, c=8: 0:SSSS 1:SSSS 2:SSSS 3:SSSS 4:SSSS 5:SSSS 6:SSSS 7:SSSS 8:SSSS 9:SSSS 10:SSSS 11:SSSS 12:SSSS 13:SSSS 14:SSSS 15:SSSS
- launch 3, repeat 3, c=4: 0:SS 1:SS 2:SS 3:SS 4:SS 5:ff 6:ff 7:ff 8:ff 9:SS 10:SS 11:SS 12:ff 13:ff 14:ff 15:ff
