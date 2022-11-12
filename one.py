from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

from pydrive.drive import GoogleDrive

drive = GoogleDrive(gauth)

csvs = []
text = []
jpgs = []
pngs = []
misc = []
pdfs = []

#Auto-iterate through all files that matches this query
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file1 in file_list:
    if(file1['title'][-3::] == "txt"):
        text.append(file1['title'])
        file = drive.CreateFile({'id': file1['id']})
        file.GetContentFile(file1['title'])
    elif(file1['title'][-3::] == "csv"):
        csvs.append(file1['title'])
        # file = drive.CreateFile({'id': file1['id']})
        # file.GetContentFile(file1['title'])
    elif(file1['title'][-3::] == "peg"):
        jpgs.append(file1['title'])
        # file = drive.CreateFile({'id': file1['id']})
        # file.GetContentFile(file1['title'])
    elif(file1['title'][-3::] == "png"):
        pngs.append(file1['title'])
        # file = drive.CreateFile({'id': file1['id']})
        # file.GetContentFile(file1['title'])
    elif(file1['title'][-3::] == "pdf"):
        pdfs.append(file1['title'])
        # file = drive.CreateFile({'id': file1['id']})
        # file.GetContentFile(file1['title'])
    else:
        misc.append(file1['title'])
        # file = drive.CreateFile({'id': file1['id']})
        # file.GetContentFile(file1['title'])


    print('title: %s, id: %s' % (file1['title'], file1['id']))


print("\nThe CSV files are:")
for i in csvs:
    print(i)
    

print("\n")

print("The TEXT files are:")
for i in text:
    print(i)

print("\n")

print("The JPEG files are:")
for i in jpgs:
    print(i)

print("\n")

print("The PNG files are:")
for i in pngs:
    print(i)

print("\n")

print("The PDF files are:")
for i in pdfs:
    print(i)

print("\n")

print("The MISC files are:")
for i in misc:
    print(i)

# file = drive.CreateFile({'id': '1KIM7h328u2octTE6GV5P_-12cnD6Dp6D'})
# file.GetContentFile('my-awesome-file.txt') 