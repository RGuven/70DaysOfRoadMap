import urllib.request as tr
import urllib
u=input("Enter the image URL: ") #http://behindwoods.com/tamil-actress/sneha/sneha-stills-photos-pictures-247.jpg
i=int(input("how many image do you want download: "))
print("Downloading started")
while i>0:
      i=i-1
      URL=u+str(i)+".jpg"
      IMAGE = URL.rsplit('/',1)[1]
      urllib.request.urlretrieve(URL, IMAGE)
      print("Downloading image",i)
