import matplotlib.pyplot as plt
import os
import cv2
import json
import skimage
import numpy as np
import requests
from datetime import date,datetime
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from lime.wrappers.scikit_image import SegmentationAlgorithm
from matplotlib.colors import LinearSegmentedColormap

def get_ImageNet_ClassLabels(json_file=False):
    ''' Input:
            json_file:  Path to JSON File, if file is already downloaded, 
                        filepath can be passed as the input parameter
            json_file:  if file is not available locally then setting False will download the file automatically.  
        Output:
            class_names: A list containing the 1000 classes names from ImageNet dataset '''
    if os.path.isfile(json_file) == False:
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        r = requests.get(url, allow_redirects=True)
        open(json_file, 'wb').write(r.content)
        print('ImageNet Classes JSON File Downloaded')
    else:
        url = json_file
#         print('JSON File Loaded')
    with open(url) as file:   
        class_names = [v[1] for v in json.load(file).values()]
    return class_names

def load_model(model_name='ResNet50'):
    if model_name=='ResNet50':
        model = ResNet50(weights='imagenet')
    if model:
        print('BlackBox Model Selected: \t\t',model_name)
        print('BlackBox Model Layers Count: \t\t',len(model.layers))
        print('BlackBox Model Weights Count: \t\t',len(model.weights))
        return model
    
        
def get_file_name(file):
    filename = file.split("//")[-1]
    filename = os.path.splitext(filename)[0]
    filename = filename.split("_")[-1]
    return filename
def read_process_image(filename,model):
    ''' Args:
           dataset_name:    Name of Dataset
           filename:        Path to Image           '''
    rows,cols = model.input.shape[1],model.input.shape[2]
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,[rows,cols])
    return img
def get_class_idx_label_score(predicted,class_names):
    ''' 
    Args:
        predicted:  Prediction returned by Blackbox model 1x1000
        model_name: Model Name to be used to make predictions
        class_names:    List of class names for the dataset
    Result:
        PDI:                Predicted Class Index
        class_probability:  class_probability of current prediction
        PDL:                Predcited Class Label
        '''
    PDI = np.argmax(predicted)
    class_probability = predicted[0][PDI]
    PDL = class_names[PDI]
    return (PDI,class_probability,PDL)
def plot_save_prediction(X_s,PDL,class_probability,result_folder,file_name,save_image=False,plot_everything=False):
    '''
    Args:   
    X_s:    Image(3d numpy array)
    PDL:    Predicted Class Label
    class_probability:  Class Probablity or Predicirton Score returned by Blackbox Model
    curr_results:       Path to save the figure
    save_image:         Plot only if False, Plot and Save if True
    Result: '''
    plt.figure(figsize=(3,3))
    plt.imshow(X_s)
    if plot_everything:
        plt.xlabel(PDL)
        plt.title(str(PDL)+':'+str(class_probability))
    # plt.title(r'$\alpha > \beta$')
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if save_image:
        plt.savefig(result_folder+'//Predicted_'+str(PDL)+'_'+str(round(class_probability,3))+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
    
def own_seg(image,md,ks,random_seed=1234,ratio=0.2,seg_algo='quickshift'):
    '''
    skimage.segmentation.quickshift(image, ratio=1.0, kernel_size=5, max_dist=10, 
    return_tree=False, sigma=0, convert2lab=True, rng=42, *, channel_axis=-1)
    '''
    ''' Function to Get Perform Segmentation using Quickshift and slic algorithms.
    Args:       
            X_s:        3d nupmy array 
            ks:         kernal  Size,float (Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters)
            md:         Max Dist, float, Cut-off point for data distances. Higher means fewer clusters
            seg_algo:   Segmentation Algorithm Name, Default: quickshift
            ratio:      float, optional, between 0 and 1
            sigma:      Width for Gaussian smoothing as preprocessing. Zero means no smoothing
    Results: 
            Segments:               Segments Created
            Segs:                   Number of Segments
            fn_segmentation:        Segmenter Function to take iamges and create segments'''
    if seg_algo == 'quickshift':
        segmenter_fn = SegmentationAlgorithm('quickshift', kernel_size=ks,max_dist=md, ratio=ratio,random_seed=random_seed)
        segments = segmenter_fn(image)
        segs = np.unique(segments).shape[0]
    elif seg_algo == 'slic':
        segmenter_fn = SegmentationAlgorithm('slic',compactness=md,max_num_iter=ks, ratio=ratio,random_seed=random_seed)
        segments = segmenter_fn(image)
        segs = np.unique(segments).shape[0]
#         print(np.max(segments))
    def fn_segmentation(image):
            return segments
    return segments,segs,fn_segmentation

def plot_seg_image(image,segments,md,ks,sub_results,file_name,save_image=False,plot_everything=True):
    '''
    Args:  
        immgg:  Input Image (3d numpy array)
        segs:   Number of segments
        md:     Max Distance (Segmentation Parameters)
        ks:     Kernel Size  (Segmentation Parameters)
        PDL:    Predicted Class Label
        class_probability:  Class Probablity or Predicirton Score returned by Blackbox Model
        file_name:          Path to save the figure
        save_image:         Plot only if False, Plot and Save if True

    Result:     
    '''
    segs = np.unique(segments).shape[0]
    immgg=skimage.segmentation.mark_boundaries(image, segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
    plt.figure(figsize=(3,3))
    plt.imshow(immgg)
    if plot_everything:
        plt.title(str(segs)+ '_'+str(md)+'_'+str(ks))
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//Segs_'+str(segs)+ '_'+str(md)+'_'+str(ks)+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
def plot_classification_score(explanation,data,labels,class_probability,sub_results,ttl,draw_quantile=False,quantile=[0.05,0.95],save_image=False,plot_points=1000,plot_everything=True):
    ''' Args:
            Explanation:        Explaination returned by Lime-Image
            data:               Data returned by LIME-Image (dense num_samples * num_superpixels)       
            labels:             Prediction Probabilities Matrix generated by LIME-Image          
            class_probability:  Class Probability or Prediction Score Returned by BlackBox Model 
            curr_results:       Path to Save the Figure 
            filenameee:         Filename to save the corrosponding Figure with into Relavant Directory      
            draw_quantile:      False, set it to True if quantile plotting on classification score is also needed 
            quantile:           Quantile Upper and Lower bound for classification score on Default [0.05-0.95]'''
    colors = ['#6d9eeb','#f9cb9c']
    cm = LinearSegmentedColormap.from_list("Custom", colors)
    x = [np.sum(d) / len(d) for d in data]
    TL = explanation.top_labels[0]
    y =labels[:,TL]
    segs = data.shape[1]
    nos = data.shape[0]
    plt.figure(figsize=(3,3))
    plt.scatter(x[:plot_points],y[:plot_points] , c =y[:plot_points] , cmap = cm, s=20 , lw = 0.5 , edgecolors = 'black')
    plt.scatter(x[0],y[0] , c='m' ,marker='x', s =200)
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axhline(class_probability, ls = '--' , lw = 2 , color ='g' )
    if draw_quantile:
        q_lower = np.quantile(y,quantile[0])
        q_upper = np.quantile(y,quantile[1])
        plt.axhline(q_lower, ls = '--' , lw = 1 , color ='red' )
        plt.axhline(q_upper, ls = '--' , lw = 1 , color ='blue' )
    if plot_everything:
        plt.text(+0.02,y[0]-0.10, '$y=f(\\xi)$' , fontsize='15')
        plt.ylabel('$f(\\xi_x)$',fontsize='15')
        plt.xlabel('$|x|$' , fontsize='15')
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//ClassScore_'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()
    

def fun_create_heatmap_lime(image,explanation,top_label,segments):
    heatmap = np.zeros_like(image[:, :, 0], dtype=np.float)
    for index, feature_importance in explanation.local_exp[top_label]:
        segm_map = segments == index
        np.putmask(heatmap, segm_map, feature_importance)
    return heatmap
def plot_heatmap_lime(heatmap,maxval,sub_results,ttl,save_result=False,show_color_bar=False,color_bar_location='right'):
    plt.figure(figsize=(3,3))
    plt.imshow(heatmap , cmap='bwr', vmin = -maxval, vmax = maxval )#, cmap='cool')
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if show_color_bar==True:
        if color_bar_location == 'right':
            plt.colorbar(im,fraction=0.046, pad=0.04, orientation='horizontal')
        if color_bar_location == 'bottom':
            plt.colorbar(im,fraction=0.046, pad=0.04, orientation='vertical')
    if save_result:
        plt.savefig(sub_results+'//Heatmap_'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()

def check_folders(path_):
    ''' Args:
    path_: Path Verifier    '''
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(path_, ' created')

def get_img_mask_lime(explanation,TL, sub_results,ttl,positive_only=True,save_image=False, num_features=200, hide_rest=False):
    ''' Fuction to Highlight Positive and Negative Features Provided by LIME-Image
    Args:   
        explanation  :   explanation computed by LIME image Module
        TL           :   Predicted Top Label by LIME-Image Explanation
        savepath     :   Path to Save the Figure
        num_features :   Features to highlight, Default: 200
    Result:
        temp         :      
        mask         :      
     '''
    temp, mask = explanation.get_image_and_mask(TL, positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
    plt.figure(figsize=(3, 3))
    plt.imshow(mark_boundaries(temp/255 / 2 + 0.4, mask))
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    plt.tight_layout()
    if save_image:
        plt.savefig(sub_results+'//ExpByLime'+ttl+'.png',transparent = True,bbox_inches = 'tight',pad_inches = 0.02, dpi = 150)
    plt.show()

def evaluate_explanation(explanation,X,all_Ys,beta,f_x,RC_Y,r2_score,data_to_csv,model_name):
    ''' 
    This function will evaluate the explanation produced by LIME-Image
    Args:
        explanation  :   explanation computed by LIME image Module 
        X            :   data returned and being used by LIME explain_instance module
        all_Ys       :   labels (for generated data) returned and being used by LIME explain_instance module
        data_to_csv  :   A Initial dictionary having keys and values to be saved in .csv file
        model_name   :   Blackbox Model Name as string being used 
        seg_range    :   Segments Range (25-50,50-100,100-150,150-200)
    Return:
        
    '''
    TL = explanation.top_labels[0]    
    Y =all_Ys[:,TL]
    maxval = np.max(np.abs(beta))
#     beta = [v for _,v in explanation.local_exp[TL]]
#     beta = np.array(beta)
    g_x = explanation.local_pred[TL][0]
    r2_score = explanation.score[TL]
    local_pred = explanation.local_pred[TL]
    intercept = explanation.intercept[TL]
#     r2_score = explanation.score[TL]

    data_to_csv['model_name']     = model_name
    data_to_csv['f_x']            = f_x
    data_to_csv['g_x']            = explanation.local_pred[TL][0]
    data_to_csv['q05_Y']          = np.quantile(Y,0.05)
    data_to_csv['q95_Y']          = np.quantile(Y,0.95)
    data_to_csv['q01_Y']          = np.quantile(Y,0.01)
    data_to_csv['q99_Y']          = np.quantile(Y,0.99)
    data_to_csv['std_Y']          = np.std(Y)
    data_to_csv['std_abs_Y']      = np.std(np.abs(Y))
    data_to_csv['r2']             = r2_score
    data_to_csv['maxval']         = maxval
    data_to_csv['local_pred']     = local_pred
    data_to_csv['intercept']      = intercept
    data_to_csv['std_beta']       = np.std(beta)
    data_to_csv['std_abs_beta']   = np.std(np.abs(beta))
    data_to_csv['mean_beta']      = np.mean(beta) 
    data_to_csv['mean_abs_beta']  = np.mean(np.abs(beta))
    data_to_csv['q05_beta']       = np.quantile(beta,0.05)
    data_to_csv['q95_beta']       = np.quantile(beta,0.95)
    data_to_csv['q01_beta']       = np.quantile(beta,0.01)
    data_to_csv['q99_beta']       = np.quantile(beta,0.99)
    data_to_csv['q25_beta']       = np.quantile(beta,0.25)
    data_to_csv['q75_beta']       = np.quantile(beta,0.75)
    data_to_csv['q10_beta']       = np.quantile(beta,0.10)
    data_to_csv['q90_beta']       = np.quantile(beta,0.90)  
    data_to_csv['max_beta']       = np.max(beta)
    data_to_csv['min_beta']       = np.min(beta)
    data_to_csv['cv_beta']        = np.std(beta) / np.mean(beta)
    data_to_csv['cv_abs_beta']    = np.std(np.abs(beta)) / np.mean(np.abs(beta))
    data_to_csv['RC_Y']           = RC_Y #(np.quantile(Y,0.99) - np.quantile(Y,0.01)) / f_x
def time_stamp():
    today = date.today()
    now = datetime.now()

    time_stamp = now.strftime("%H%M%S")+'_'+today.strftime("%d%m%Y")
    return time_stamp
#######################################     SEGMENTATION FUNCTIONS    #########################################
def get_segment_number(image, md,ks,seg_algo,random_seed=1234,ratio=0.2):
    ''' COMPUTE NO OF SEGMENTS by using hyperparametrs
    Args:
        image:       Image to be segmented (MxNx3) 
        md:          Max Distance
        ks:          Kernel Size
        seg_algo:    Algorithm used for segmentation '''
    segmentation_fn = SegmentationAlgorithm(seg_algo, kernel_size=ks, max_dist=md, ratio=ratio, random_seed=random_seed) 
    segments = segmentation_fn(image)
    return len(np.unique(segments))
def search_segment_number(image, target_seg_no, init_max_dist=100,init_kernel_size=4,seg_algo='quickshift'):
    ''' search_segment_number by implementing dichotomic_search
    Args:
    image:                Image to be segmented (MxNx3) 
    target_seg_no:        Target Segments Number       
    init_max_dist:        Initial Max Distance used for creating segments     
    init_kernel_size:     Initial Kernel Size used for creating segments    
    seg_algo:             Algorithm used for segmentation          
    Return:
    rmd:Max Distance Required to create Target Segments Number
    init_kernel_size:     Kernel Size Required to create Target Segments Number 
    '''
    random_seed=1234
    ratio=0.2
    lmd, rmd,ks = 0, init_max_dist,init_kernel_size
    lsn = get_segment_number(image, lmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
    rsn = get_segment_number(image, rmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
    niter = 0
    while niter<20 and rsn!=target_seg_no:
        niter += 1
        mmd = (lmd + rmd) / 2.0
        msn = get_segment_number(image, mmd,ks,seg_algo,random_seed=random_seed,ratio=ratio)
#         print(f'{lmd}:{lsn}  {mmd}:{msn} {rmd}:{rsn}')
        if msn <= target_seg_no <= lsn:
            rsn, rmd = msn, mmd
        else:
            lsn, lmd = msn, mmd
    return rmd,init_kernel_size,random_seed,ratio
# def merge_two_dicts(x, y):
#     """Given two dictionaries, merge them into a new dict as a shallow copy."""
#     z = x.copy()
#     z.update(y)
    return z
def segs_sections(segs,seg_list):
        for sl in  range(0,len(seg_list),1):
            if segs >= seg_list[sl][0] and segs <=seg_list[sl][1]:
                return [seg_list[sl][0],seg_list[sl][1]]

def get_beta_from_expl(expl):
    n = len(np.unique(expl.segments))
    beta = np.zeros(n)
    for i,v in expl.local_exp[ expl.top_labels[0] ]:
        beta[i] = v
    return beta
def get_RCY(Y,class_prob):
    return (np.quantile(Y,0.99) - np.quantile(Y,0.01)) / class_prob
    
def heatmap_from_beta(segments, beta):
    heatmap = np.zeros_like(segments, dtype=np.float32)
    for segm, importance in enumerate(beta):
        heatmap[ segments==segm ] += importance 
    return heatmap
