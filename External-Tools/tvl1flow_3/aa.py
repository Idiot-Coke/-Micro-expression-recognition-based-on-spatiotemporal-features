import os

path = './a/' 
png_path = './b/'
length = len(os.listdir(path))
num = 1
while(num<length):
   ml = './color_flow\t' + path + str(num) + '.flo\t' + png_path + str(num) + '.png'
   os.system(ml)
   num += 1
