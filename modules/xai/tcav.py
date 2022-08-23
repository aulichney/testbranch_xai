from pathlib import Path
from .utils import encode_folder
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, ttest_1samp

def tcav(model, transforms, layer, concept_path, random_path, class_path, results_path, class_idx, samples=50, n=50, device="cpu", batch_size=8, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    elif isinstance(rng, int): rng = np.random.default_rng(rng)
    available_images = len([p for p in Path(concept_path).glob("*") if p.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]])
    # Encode
    random_acts_list, random_grads_list = encode_folder(model, random_path, random_path, layer, class_idx, transforms, workers=1, batch_size=batch_size, device=device, namestring=f"_{Path(random_path).name}", limit=-1)
    random_acts_list, random_grads_list = random_acts_list.reshape(random_acts_list.shape[0],-1), random_grads_list.reshape(random_grads_list.shape[0],-1)
    concept_acts_list, concept_grads_list = encode_folder(model, concept_path, concept_path, layer, class_idx, transforms, workers=1, batch_size=batch_size, device=device, namestring=f"_{Path(concept_path).name}", limit=-1)
    concept_acts_list, concept_grads_list = concept_acts_list.reshape(concept_acts_list.shape[0],-1), concept_grads_list.reshape(concept_grads_list.shape[0],-1)
    class_acts_list, class_grads_list = encode_folder(model, class_path, results_path, layer, class_idx, transforms, workers=1, batch_size=batch_size, device=device, namestring=f"_{class_idx}", limit=-1)
    class_acts_list, class_grads_list = class_acts_list.reshape(class_acts_list.shape[0],-1), class_grads_list.reshape(class_grads_list.shape[0],-1)
    # iterate
    cavs, scores, r_scores, pvals, accs = [], [], [], [], []
    for i in range(n):
        # sample
        pos_activations = rng.choice(concept_acts_list, min(samples,len(concept_acts_list)), replace=False)
        rand_activations = rng.choice(random_acts_list, min(samples,len(random_acts_list)), replace=False)
        rand_activations2 = rng.choice(random_acts_list, min(samples,len(random_acts_list)), replace=False)
        # get cav
        cav, acc = get_cav_from_activations(pos_activations,rand_activations)
        # test cav
        score = tcav_from_gradients(class_grads_list,cav)
        # get rcav
        r_cav, r_acc = get_cav_from_activations(rand_activations2,rand_activations)
        # test rcav
        r_score = tcav_from_gradients(class_grads_list, r_cav)
        # append
        cavs.append(cav)
        scores.append(score)
        r_scores.append(r_score)
        pvals.append(cav)
        accs.append(acc)
    # significance
    t_val, p_val = ttest_ind(scores, r_scores)
    #t_val, p_val = ttest_1samp(scores, 0.5)
    # summarize
    cav = np.mean(cavs, axis=0)
    score_mean = np.mean(scores, axis=0)
    score_std = np.std(scores, axis=0)
    pval = p_val
    acc = np.mean(accs, axis=0)
    return cav.tolist(), score_mean, score_std, pval, acc, available_images

def get_cav_from_activations(pos_activations,rand_activations):
    # split data
    x = np.concatenate((pos_activations,rand_activations))
    y = [1]*len(pos_activations)+[0]*len(rand_activations)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y)
    # train linear model
    lm = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    lm.fit(x_train, y_train)
    #validate model
    y_pred = lm.predict(x_test)
    acc = metrics.accuracy_score(y_pred, y_test)
    return lm.coef_[0], acc

def tcav_from_gradients(classes_gradients,cav):
    sensitivities = np.dot(np.array(classes_gradients), np.array(cav))
    # then, the tcav score it's easier to compute
    scores=(sensitivities>0).sum()/len(sensitivities)
    return scores
