def find_se(per, las):
    pers = per.split('/')[1:]
    # print(pers)
    lass = las.split('/')[1:]
    # print(lass)
    for i in range(len(pers)):
        for j in range(len(lass)):
            if pers[i]==lass[j]:
                return pers[i]
    return "not find"

def cutting(per, persub):
    return per[:per.find(persub)-1]

def find_se_com(per,las):
    persub = find_se(per,las)
    return cutting(per,persub) + las

if __name__ == '__main__':
    persub = (find_se("/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static",
                   "/algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG"))
    p = cutting("/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static", persub)

    persub1 = (find_se_com("/Users/aldno/Downloads/flaskDemo-master/algorithm/cutimg/static",
                      "/algorithm/cutimg/static/images_GLCM_original/images_camouflage/mix/20m/2.JPG"))
    print(persub1)