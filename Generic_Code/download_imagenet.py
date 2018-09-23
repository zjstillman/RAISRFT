import urllib.request
import urllib
import os

file = open("fall11_urls.txt", "r")
i = 0
while i < 10:
	try:
		f = open('imagenet/' + str(i) + '.jpg','wb')
		f.write(urllib.request.urlopen(file.readline().split()[1]).read())
		f.close()
		statinfo = os.stat('imagenet/' + str(i) + '.jpg')
		if statinfo.st_size > 0:
			i += 1
		else:
			print('Empty: ' + str(i))
	except:
		print('Failed: ' + str(i))