
import os

jpg_file=[]
xml_file=[]

for filename in os.listdir(os.chdir("YOUR PATH")):

    if filename.endswith(".xml"):
        xml_file.append(filename)
    elif filename.endswith(".jpg"):
        jpg_file.append(filename)


counter=0
for jpg in jpg_file:
    counter+=1
    for xml in xml_file:
        if jpg[:-4]==xml[:-4]:
            os.rename(jpg,str(counter)+"_patatoes.jpg")
            os.rename(xml,str(counter)+"_patatoes.xml")
