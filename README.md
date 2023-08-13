# diffusion_cross_modal_retrieval
test_img_fid.py: 计算fid score，以及real(fake) image和相对应promote的clip相似度分数，结果保存在result，如：

121.0 23.8027 28.2696

.....

121.0 表示两个图像的fid score, 23.8027表示real image和promot的clip score, 28.2696表示fake image和prompt的clip score

test_img1.py: pretrained stable diffusion直接在flickr30k测试集生成的图像，保存为Generated_images，相对应的真值图像在original_images

eval_finetuning_contrastive.py: 评估检索分数

test_caps.txt: 测试集的promots, 一张图像有5个prompt

 


