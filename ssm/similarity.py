from sewar.full_ref import ssim
import cv2,os
t = []
for i in os.listdir("./Testing"):
    testing_img = cv2.imread("./Testing/"+i)
    best_val = 0
    best_name = ""
    for j in os.listdir("../src/Gait Energy Image/GEI"):
        for k in os.listdir("../src/Gait Energy Image/GEI/"+j):
            trial_img = cv2.imread("../src/Gait Energy Image/GEI/"+j+"/"+k)
            s = sum(ssim(trial_img,testing_img))/2
            print(str(i).split("_")[0]+"\t---\t"+str(j)+"\t--> ",s)
            if s>best_val:
                best_val = s
                best_name = j
    t.append((i.split("_")[0],best_name,best_val))
print("\n\nTested\t\tPredicted\t\tDistance")
acc=0
for i in t:
    print(i[0],"\t\t",i[1],"\t\t",i[2])
    if(i[0]==i[1]):
        acc+=1
print("Accuracy:",str((acc/17)*100))