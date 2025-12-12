import os
import img2pdf
import argparse
os.system("ls -d LEC* > .list")
parser = argparse.ArgumentParser()
parser.add_argument("-s", type=int, default=1)
args = parser.parse_args()
if os.path.exists(".list"):
    dirs = []
    with open(".list","r") as f:
        contents = f.read()
        dirs = contents[:-1].split('\n')
    print(dirs)
    for dir in dirs:
        lec = ''
        for i in dir[::-1]:
            if i.isnumeric():
                lec+=i
            else:
                break
        lec = lec[::-1]
        if len(lec)!=2:
            lec = '0'+lec
        if len(lec) < 2:
            lec = '0'+lec
        os.system("cd "+dir+"; ls > .list")
        pages = []
        with open(dir+"/.list","r") as f:
            contents = f.read()
            pages = list(map(lambda x: dir+'/'+x,contents[:-1].split('\n')))
        print(pages)
        with open(lec+".pdf", "wb") as f:
            f.write(img2pdf.convert(pages))
        os.system("rm -R "+dir)
os.system("rm .list")
