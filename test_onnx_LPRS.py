import onnxruntime as ort
import pickle
import numpy as np
import time
import cv2

import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import sys
import torch.nn.functional as F
#from utils import *
from transformers import PreTrainedTokenizerFast
import sympy


LPR_W = 192
LPR_H = 128

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ]
)

tokenizerfile = 'ONNX_LPR/tokeniserLPR.json'

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizerfile)
vocab = tokenizer.get_vocab()
print('Vocab SIze:',len(list(vocab.keys())))

def token2str(tokens, tokenizer):
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [detok.replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]


def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def image_preprocess(image,fix_dim = (64,480),return_re=0): #(H,W)
    ih, iw    = fix_dim
    dim  = image.shape
    gray = 0
    if len(dim) == 2:
        gray = 1
                
    print("value of gray is   ", gray)
    h,w = dim[0],dim[1]
    #inter = cv2.INTER_CUBIC
    #inter = cv2.INTER_LINEAR
    inter = cv2.INTER_AREA
    args_dict = {}
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh), interpolation = inter)

    if return_re:
        return image_resized,''

    if not gray:
        image_paded = np.full(shape=[ih, iw, 3], fill_value=255.0,dtype = image_resized.dtype)
    else:
        image_paded = np.full(shape=[ih, iw], fill_value=255.0,dtype = image_resized.dtype)
    #image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    if not gray:
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    else:
        image_paded[dh:nh+dh, dw:nw+dw] = image_resized
    #image_paded[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
    image_paded = np.array(image_paded,dtype = image_resized.dtype)
    #args_dict['yindex'] = dh
    #args_dict['xindex'] = dw
    #args_dict['y2index'] = nh+dh
    #args_dict['x2index'] = nw+dw
    #args_dict['resizex'] = nw
    #args_dict['resizey'] = nh
    args_dict['originalsize'] = (h,w)
    args_dict['resize'] = (nh,nw)
    args_dict['targetsize'] = fix_dim
        
    return image_paded,args_dict



device = 'cpu' #'cuda'
dec = torch.jit.load('./jit_models/decoder_LPR_CPU_FULL.pt')
dec.eval()
dec.to(device)


backbone_file = './ONNX_LPR/BACKBONE_CPU_LPR.onnx'
encoder_file = './ONNX_LPR/ENCODER_CPU_LPR.onnx'

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
#sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

#m = ort.InferenceSession('./RSN18_CUDA_DYN_SIMP.onnx')
print("Loading backbone model...")
mback = ort.InferenceSession(backbone_file, sess_options, providers=['CPUExecutionProvider'])
print("Backbone Model is loaded")

print("Loading Encoder model...")
menc = ort.InferenceSession(encoder_file, sess_options, providers=['CPUExecutionProvider'])
print("Encoder Model is loaded")

inputs = mback.get_inputs()
outputs = mback.get_outputs()
inputs = [(x.name,x.type,x.shape) for x in inputs]
outputs = [(x.name,x.type,x.shape) for x in outputs]


def image_preprocess(image,fix_dim = (64,480),return_re=0): #(H,W)
    ih, iw    = fix_dim
    dim  = image.shape
    gray = 0
    if len(dim) == 2:
        gray = 1
                
    h,w = dim[0],dim[1]
    #gh = int(0.5*h)
    #gw = int(0.5*w)
    #image = cv2.resize(image, (gw, gh), cv2.INTER_LINEAR)

    #inter = cv2.INTER_CUBIC
    #inter = cv2.INTER_LINEAR
    inter = cv2.INTER_AREA
    args_dict = {}
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh), interpolation = inter)

    if return_re:
        return image_resized,''

    if not gray:
        image_paded = np.full(shape=[ih, iw, 3], fill_value=255.0,dtype = image_resized.dtype)
    else:
        image_paded = np.full(shape=[ih, iw], fill_value=255.0,dtype = image_resized.dtype)
    #image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    if not gray:
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    else:
        image_paded[dh:nh+dh, dw:nw+dw] = image_resized
    #image_paded[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
    image_paded = np.array(image_paded,dtype = image_resized.dtype)
    #args_dict['yindex'] = dh
    #args_dict['xindex'] = dw
    #args_dict['y2index'] = nh+dh
    #args_dict['x2index'] = nw+dw
    #args_dict['resizex'] = nw
    #args_dict['resizey'] = nh
    args_dict['originalsize'] = (h,w)
    args_dict['resize'] = (nh,nw)
    args_dict['targetsize'] = fix_dim
        
    return image_paded,args_dict

def PlateOCR(inpImage):
   #cv2.imwrite("lastplate.png", inpImage)
   #inpImage = cv2.imread("lastplate.png")
   imgh, imgw, _ = inpImage.shape
   i1 = image_preprocess(inpImage,fix_dim = (LPR_H,LPR_W),return_re=0)[0]
  
   #i1 = cv2.imread('paddedlicPlate.png')
   #i1 = cv2.resize(i1,(192,128))
   i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY ) 

   vis2 = cv2.cvtColor(i1, cv2.COLOR_GRAY2BGR)
   #cv2.imshow("scaledplate", vis2)
   #cv2.waitKey(0)
   i1 = i1[None,None,:,:]
   i1 = i1.astype(np.float32)
   i1 = i1/255
   i1 -= 0.7931
   i1 = i1/0.1738 
   #print(i1.shape)

   #i1 = test_transform(image=i1)['image'][:1].unsqueeze(0)
   
   
   for i in range(1):
       t1 = time.time()
       outputs = mback.run(None, {'x.1': i1})
       #print(outputs[0])
       t2 = time.time()
       #print('Backbone Time ms: ',(t2-t1)*1000)
    
       t1 = time.time()
       outputenc = menc.run(None, {'x.1': outputs[0]})
       #print('outputenc[0]', outputenc[0])
       t2 = time.time()
       #print('Encoder Time ms: ',(t2-t1)*1000)
       
       ############################DECODER################################
       #print(type(outputenc[0]))
       encoded = torch.from_numpy(outputenc[0])
       #print(type(encoded))
       #print("encoded: ", encoded)
       t1 = time.time()
       start_tokens = torch.LongTensor([1]*len(encoded))[:, None].to(device)
       #print("start_tokens shape is", start_tokens.shape, start_tokens)

       num_dims = len(start_tokens.shape)
       if num_dims == 1:
          start_tokens = start_tokens[None, :]
       b, t = start_tokens.shape
       out = start_tokens
       mask = torch.full_like(out, True, dtype=torch.bool).to(device)
  
       max_seq_len = 25#300
       eos_token = 2
       filter_thres = .9
       #print("b and t is ",b,t)
       #print("mask is ",mask)
       ocroutconf = 0
       ocroutminconf = 1.0
   
       for _ in range(max_seq_len):
	   #x = out[:, -max_seq_len:]
	   #mask = mask[:, -max_seq_len:]
           x = out
           mask = mask
           #print(f'{_}: | x:{x.shape} | mask:{mask.shape}|')
           #print("x is ", x, ", mask ", mask)
           with torch.no_grad():
	       #logits = dec(x,mask,{'context': encoded})
               logits = dec(x,mask,encoded)
	       #print('Logits:',logits.shape)
               logits = logits[:,-1,:]
	   #print("Logits: ", logits)
	   #print('Logits from Decoder:',logits.shape,logits[0,:5],logits[0,-5:])

           logits = F.softmax(logits, dim=-1)

           maxconf,maxind  = torch.max(logits,dim=1)
           print('maxid:',maxind,', conf: ', maxconf.item())
           out = torch.cat((out, maxind[:,None]), dim=-1)
           ocroutconf = ocroutconf + maxconf.item()
           ocroutminconf = min(ocroutminconf, maxconf.item())
    
           mask = F.pad(mask, (0, 1), value=True)
	   #print('eos_token',eos_token)
           if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
               break
	   #print('=='*10,'\n')
	   
       out = out[:, t:]
       if num_dims == 1:
           out = out.squeeze(0)
       t2 = time.time()
       ############################DECODER################################
       #tok_avg+=out.shape[-1]*out.shape[0]
       #tok_avg+=dec.shape[-1]
       ocroutconf = ocroutconf / len(out[0])
       pred = token2str(out, tokenizer)[0]
     
       #print('Logits:',logits.shape)
       #print('dec',out[:,:5])
       #print("LPR output: ", pred, ', Conf: ', ocroutconf)
       #print(f'Output from Decoder {out.shape} | Generation Time {(t2-t1):.3f} (in sec)')
       return pred, ocroutconf, ocroutminconf
	   

#image = cv2.imread("licenceplate.png")
#PlateOCR(image)    
    
