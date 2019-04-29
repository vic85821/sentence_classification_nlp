wget -O ./models/bert/ckpts/random-8981_epoch-9.ckpt "https://gntuedutw-my.sharepoint.com/:u:/g/personal/r07922058_g_ntu_edu_tw/ESvUJcD632JDoz7MA4UbTyUB3jP505acGewwa4HLkKYBcQ?download=1"

cp ./models/bert/ckpts/random-8981_epoch-9.ckpt ./models/ensemble/ckpts

wget -O ./models/ensemble/ckpts/random-3319_epoch-6.ckpt "https://gntuedutw-my.sharepoint.com/:u:/g/personal/r07922058_g_ntu_edu_tw/EYxMDniLVZxFsKRsdONDTKIBtHsS3Mf_9yIJGp1WAuwD1Q?download=1"

wget -O ./models/ensemble/ckpts/random-651_epoch-5.ckpt "https://gntuedutw-my.sharepoint.com/:u:/g/personal/r07922058_g_ntu_edu_tw/ERDJ8EzAhNFBuDxAVkWirEMBUROPC89q7UtwBosk-gK_Pg?download=1"
