import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
# from skdim.id import MLE
from GPTID.IntrinsicDimCUDA import PHD, PH
from GPTID.IntrinsicDimCUDA import gpu_dist_matrix


# Insert here path to model files in your syste,
model_path = 'Qwen/Qwen1.5-1.8B'
tokenizer_path = model_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)


"""
Our method (PHD) is stochastic, here are some magic constants for it. They are chosen specifically for text data. If you plan to use this code for something different, consider testing other values.

MIN_SUBSAMPLE       --- the size of the minimal subsample to be drawn in procedure. Lesser values yields less statisitcally stable predictions.
INTERMEDIATE_POINTS --- number of sumsamples to be drawn. The more this number is, the more stable dimension estimation for single text is; however,  the computational time is higher, too. 7 is, empirically, the best trade-off.
"""
MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7


'''
Auxillary function. Clear text from linebreaks and odd whitespaces, because they seem to interfer with LM quite a lot.
Replace with a more sophisticated cleaner, if needed.
'''


def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')


def decode_by_tokens(inputs):
    decoded_tokens = []
    for token in inputs['input_ids'].reshape(-1):
        decoded_tokens.append(tokenizer.decode([token]))

    return decoded_tokens


def get_embeds(text, model=None, tokenizer=None, returns_tokenized=False, max_length=8192):
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=max_length, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outp = model(**inputs)

    if not returns_tokenized:
        return outp[0][0].cpu().numpy()

    return outp[0][0].cpu().numpy(), decode_by_tokens(inputs)


'''
Get PHD for one text
Parameters:
        text  --- text
        solver --- PHD computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Persistence Homology Dimension method.'''


def get_phd_single(text, solver, max_length=8192):
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outp = model(**inputs)

    mx_points = inputs['input_ids'].shape[1]
    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

    print(
        "mn_points =", mn_points,
        "max_points =", mx_points,
        "point_jump =", step
    )

    print("input_shape:", outp[0][0].cpu().numpy().shape)
    dist_matrix = gpu_dist_matrix(outp[0][0].cpu().numpy())

    return solver.fit_transform(
        dist_matrix,
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step,
        dist=True
    )


def get_ph_single(text, solver, max_length=8192):
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outp = model(**inputs)

    mx_points = inputs['input_ids'].shape[1]
    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

#     print(
#         "mn_points =", mn_points,
#         "max_points =", mx_points,
#         "point_jump =", step
#     )

    print("input_shape:", outp[0][0].cpu().numpy().shape)
    dist_matrix = gpu_dist_matrix(outp[0][0].cpu().numpy())
#     print("dist_matrix.shape:", dist_matrix.shape)

    return solver.fit_transform(
       dist_matrix,
       dist=True
    )



def get_ph_single_loop(text, solver, n_tries=10, max_length=8192):
    values = []
    for _ in range(n_tries):
        values.append(get_ph_single(text, solver, max_length=max_length))
    return np.mean(values)


def get_phd_single_loop(text, solver, n_tries=10, max_length=8192):
    values = []
    for _ in range(n_tries):
        values.append(get_phd_single(text, solver, max_length=max_length))
    return np.mean(values)


def get_raw_phd_in_loop(points, alpha=1.0, n_tries=10):
    values = []
    PHD_solver = PHD(metric='euclidean',
                     n_points=9,
                     alpha=alpha)
    for _ in range(n_tries):
        values.append(get_raw_phd(points))

    return values


'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)

        alpha --- Parameter alpha for PHD computattion

Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Persistence Homology Dimension method.
'''


def get_ph(
    df,
    key='text',
    is_list=False,
    alpha=1.0,
    regression_type='vanilla',
    n_tries=1
):
    dims = []
    PH_solver = PH(alpha=alpha, metric='euclidean')
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
#         print("text ===============:", text)
        dims.append(
            get_ph_single_loop(text, PH_solver, n_tries=n_tries)
        )

    return np.array(dims).reshape(-1, 1)



def get_phd(
    df,
    key='text',
    is_list=False,
    alpha=1.0,
    regression_type='vanilla',
    n_tries=10
):
    dims = []
    PHD_solver = PHD(alpha=alpha, metric='euclidean',
                     n_points=9)
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
#         print("text ===============:", text)
        dims.append(
            get_phd_single_loop(text, PHD_solver, n_tries=n_tries)
        )

    return np.array(dims).reshape(-1, 1)


def get_raw_phd(points, alpha=1.0):
    points = points.T
    mx_points = points.shape[1]

    mn_points = MIN_SUBSAMPLE
    step = (mx_points - mn_points) // INTERMEDIATE_POINTS

    solver = PHD(
        alpha=alpha,
        metric='euclidean',
        n_points=9
    )

    print(
        "mn_points = ", mn_points,
        'max_points = ', mx_points,
        'point_jump = ', step
    )
    print("input_shape:" , points.T.shape)

    return solver.fit_transform(
        points.T,
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step
    )


'''
Get MLE for one text
Parameters:
        text  --- text
        solver --- MLE computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Maximum Likelihood Estimation method.'''


def get_mle_single(text, solver):
    inputs = tokenizer(preprocess_text(text), truncation=True,
                       max_length=2048, return_tensors="pt").to(device)
    with torch.no_grad():
        outp = model(**inputs)

    return solver.fit_transform(outp[0][0].cpu().numpy())


'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)
        
Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Maximum Likelihood Estimation method.
'''


def get_mle(df, key='text', is_list=False):
    dims = []
    MLE_solver = MLE()
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
        print(text)
        dims.append(get_mle_single(text, MLE_solver))

    return np.array(dims).reshape(-1, 1)


def sample_dims(text, n_tries, regression_type='huber'):
    dims = []
    alpha = 1.0
    PHD_solver = PHD(alpha=alpha, metric='euclidean',
                     n_points=9, regression_type=regression_type)
    for _ in tqdm(range(n_tries)):
        dims.append(get_phd_single(text, PHD_solver))
    return dims
