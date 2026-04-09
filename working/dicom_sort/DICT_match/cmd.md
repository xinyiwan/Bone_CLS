### DICOM to niftti
docker run --rm -it \
  -v /home/ext_xinwan/Bone_AI/Bone_CLS/working/dicom_sort/DICT_match:/working \
  -v /home/ext_xinwan/Bone_AI/tmp_data/BONE-AI/ADQUISICIONES:/data \
  -v /home/ext_xinwan/Bone_AI/tmp_data_nifti/ADQUISICIONES:/results \
  --entrypoint bash \
  testp11:xw 

  python /working/dcm2nifti.py

  
### change mod
docker run --rm -it \
  -v /home/ext_xinwan/Bone_AI/tmp_data_nifti:/data \
  --entrypoint bash \
  testp11:xw 

chmod -R 777 /data
