import numpy as np
import wandb
import pandas as pd
import os
from scipy.ndimage import label
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score


def calculate_metrics(preds, gts, metrics, dataset=None):
        """Calculate metrics based on predictions and ground truths using sklearn."""
        if len(preds) != len(gts):
            raise ValueError("Length of predictions and ground truths do not match.")

        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(gts, preds)
            elif metric == "precision":
                results[metric] = precision_score(gts, preds, average='macro', zero_division=0)
            elif metric == "recall":
                results[metric] = recall_score(gts, preds, average='macro', zero_division=0)
            elif metric == "jaccard":
                results[metric] = jaccard_score(gts, preds, average='macro')
            elif metric == "f1_score":
                results[metric] = f1_score(gts, preds, average='macro')

        return results



def relaxed_evaluation_Cholec80_video(predLabelID, gtLabelID, fps=1):
    """
    Evalúa el desempeño de un modelo de reconocimiento de fases quirúrgicas.

    Parámetros:
        gtLabelID (np.ndarray): Arreglo 1D con los labels verdaderos (ground truth).
        predLabelID (np.ndarray): Arreglo 1D con los labels predichos.
        fps (int): Número de frames por segundo del video.

    Retorna:
        res (list): Jaccard index por fase (con relaxed boundary), NaN si no aparece en GT.
        prec (list): Precisión por fase (con relaxed boundary).
        rec (list): Recall por fase (con relaxed boundary).
        acc (float): Exactitud total del video (con relaxed boundary).
    """

    
    gtLabelID += 1
    predLabelID += 1
    
    oriT = 10 * fps  # Duración de la ventana relajada (10 segundos en frames)

    res = []
    prec = []
    rec = []
    diff = predLabelID - gtLabelID  # Diferencia entre predicción y GT
    #diff = diff[0] # remove innecesary dimension
    updatedDiff = np.zeros(diff.shape)  # Se irá modificando con reglas de relajación

    nPhases = 7
    
    # Aplicar relaxed boundary para cada fase en el GT
    for iPhase in range(1, nPhases + 1):
        # Encontrar regiones conectadas donde la fase actual está presente en el GT
        mask_gt = (gtLabelID == iPhase).astype(int)
        labeled_gt, num_objects = label(mask_gt)
        
        # Iteramos sobre cada una de esas regiones conectadas
        for conn_id in range(1, num_objects + 1):
            # Encuentro la primer region de la fase en el Gt
            indices = np.where(labeled_gt == conn_id)[-1]
            startIdx = indices.min()
            endIdx = indices.max()

            curDiff = diff[startIdx:endIdx + 1].copy()

            # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
            t = min(oriT, len(curDiff))

            # Aplicar reglas de transición relajada, específicas por fase
            if iPhase in [4, 5]:
                # Working
                mask_late = curDiff[:t] == -1
                curDiff[np.where(mask_late)[0]] = 0 # late transition

                last_t = curDiff[-t:]  # últimos t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # early transition


            elif iPhase in [6, 7]:
                # Working
                first_t = curDiff[:t]  # primeros t elementos
                mask_last = (first_t == -1) | (first_t == -2)
                curDiff[:t][mask_last] = 0 # late transition

                last_t = curDiff[-t:]  # primeros t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # late transition

            else:
                # Working
                mask_first_t = curDiff[:t] == -1
                curDiff[np.where(mask_first_t)[0]] = 0 # late transition
                curDiff[:t][curDiff[-t:] == 1] = 0 # early transition

            updatedDiff[startIdx:endIdx + 1] = curDiff
    
    
    # Cálculo de Jaccard, Precisión y Recall por fase
    for iPhase in range(1, nPhases + 1):
        mask_gt = (gtLabelID == iPhase).astype(int)
        mask_pred = (predLabelID == iPhase).astype(int)

        # Regiones conectadas para la fase actual
        _, gt_objects = label(mask_gt)
        _, pred_objects = label(mask_pred)

        if gt_objects == 0:
            # Si no hay presencia de la fase en GT, se asigna NaN
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue

        union_idx = np.where((gtLabelID == iPhase) | (predLabelID == iPhase))[-1]
        tp = np.sum(updatedDiff[union_idx] == 0)  # true positives relajados

        # Jaccard Index
        jaccard = tp / len(union_idx) * 100
        res.append(jaccard)
        
        # Precision
        pred_sum = np.sum(predLabelID == iPhase)
        prec.append(tp * 100 / (pred_sum + 0.0000000001))

        # Recall
        gt_sum = np.sum(gtLabelID == iPhase)
        rec.append(tp * 100 / gt_sum)

    # Exactitud total
    acc = np.sum(updatedDiff == 0) / len(gtLabelID) * 100


    return acc, prec, rec, res



def relaxed_evaluation_Cholec80(dataset_preds, dataset_gts, names):
    results = {}
    video_names = list(dict.fromkeys(names))


    dataset_info = {'Video_id': names, 'Pred': dataset_preds, 'GT': dataset_gts}
    df = pd.DataFrame(dataset_info)
    video_results = []
    
    jaccard = np.zeros((40, 7))
    precision = np.zeros((40, 7))
    recall = np.zeros((40, 7))
    accuracy = np.zeros(40)

    # Iterate through each video to calculate metrics
    for idx, video in enumerate(video_names):
        filtered_df = df[df['Video_id'] == video]
        video_preds = filtered_df['Pred'].tolist()
        video_gts = filtered_df['GT'].tolist()

        if video_preds and video_gts:
            video_results.append(calculate_metrics(video_preds, video_gts, ["accuracy", "precision", "recall", 'jaccard', 'f1_score']))

            acc, pre, rec, jac = relaxed_evaluation_Cholec80_video(np.array(video_preds), np.array(video_gts))

            accuracy[idx] = acc
            precision[idx, :] = pre
            recall[idx, :] = rec
            jaccard[idx, :] = jac


        else:
            raise ValueError(f"No predictions or ground truths found for video {video} in dataset.")
    
    metrics_mean = {metric: np.mean([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'jaccard', 'f1_score']}
    metrics_std = {metric: np.std([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'jaccard', 'f1_score']}

    # Relaxed metrics adaptations
    # Limitar valores máximos a 100
    jaccard = np.clip(jaccard, None, 100)
    precision = np.clip(precision, None, 100)
    recall = np.clip(recall, None, 100)

    mean_jacc_per_phase = np.nanmean(jaccard, axis=0)
    mean_prec_per_phase = np.nanmean(precision, axis=0)
    mean_rec_per_phase = np.nanmean(recall, axis=0)

    mean_jaccard, std_jaccard = np.mean(mean_jacc_per_phase), np.std(mean_jacc_per_phase)
    mean_precision, std_precision = np.mean(mean_prec_per_phase), np.std(mean_prec_per_phase)
    mean_recall, std_recall = np.mean(mean_rec_per_phase), np.std(mean_rec_per_phase)
    mean_accuracy, std_accuracy = np.mean(accuracy), np.std(accuracy)

    metrics_mean['relaxed accuracy'] = mean_accuracy
    metrics_mean['relaxed precision'] = mean_precision
    metrics_mean['relaxed recall'] = mean_recall
    metrics_mean['relaxed jaccard'] = mean_jaccard

    metrics_std['relaxed accuracy'] = std_accuracy
    metrics_std['relaxed precision'] = std_precision
    metrics_std['relaxed recall'] = std_recall
    metrics_std['relaxed jaccard'] = std_jaccard


    results['Cholec80'] = {"mean": metrics_mean, "std": metrics_std}

    return results


def relaxed_evaluation_M2CAI_video(predLabelID, gtLabelID, fps=1):
        """
        Evalúa el desempeño de un modelo de reconocimiento de fases quirúrgicas.

        Parámetros:
            gtLabelID (np.ndarray): Arreglo 1D con los labels verdaderos (ground truth).
            predLabelID (np.ndarray): Arreglo 1D con los labels predichos.
            fps (int): Número de frames por segundo del video.

        Retorna:
            res (list): Jaccard index por fase (con relaxed boundary), NaN si no aparece en GT.
            prec (list): Precisión por fase (con relaxed boundary).
            rec (list): Recall por fase (con relaxed boundary).
            acc (float): Exactitud total del video (con relaxed boundary).
        """

        gtLabelID += 1
        predLabelID += 1
        
        oriT = 10 * fps  # Duración de la ventana relajada (10 segundos en frames)

        res = []
        prec = []
        rec = []
        diff = predLabelID - gtLabelID  # Diferencia entre predicción y GT
        #diff = diff[0] # remove innecesary dimension
        updatedDiff = np.zeros(diff.shape)  # Se irá modificando con reglas de relajación

        nPhases = 8

        for iPhase in range(1, nPhases + 1):
            # Encontrar regiones conectadas donde la fase actual está presente en el GT
            mask_gt = (gtLabelID == iPhase).astype(int)
            labeled_gt, num_objects = label(mask_gt)

            # Iteramos sobre cada una de esas regiones conectadas
            for conn_id in range(1, num_objects + 1):
                #Encuentro la primer region de la fase en el Gt
                indices = np.where(labeled_gt == conn_id)[-1]
                startIdx = indices.min()
                endIdx = indices.max()

                curDiff = diff[startIdx:endIdx + 1].copy()

                # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
                t = min(oriT, len(curDiff))

                # Aplicar reglas de transición relajada, específicas por fase
                if iPhase in [5, 6]:
                    #Working
                    mask_late = curDiff[:t] == -1
                    curDiff[np.where(mask_late)[0]] = 0 # late transition

                    last_t = curDiff[-t:]  # últimos t elementos
                    mask_early = (last_t == 1) | (last_t == 2)
                    curDiff[:t][mask_early] = 0 # early transition
                
                elif iPhase in [7, 8]:
                    # Working
                    first_t = curDiff[:t]  # primeros t elementos
                    mask_last = (first_t == -1) | (first_t == -2)
                    curDiff[:t][mask_last] = 0 # late transition

                    last_t = curDiff[-t:]  # primeros t elementos
                    mask_early = (last_t == 1) | (last_t == 2)
                    curDiff[:t][mask_early] = 0 # late transition
                
                else:
                    #Working
                    mask_first_t = curDiff[:t] == -1
                    curDiff[np.where(mask_first_t)[0]] = 0 # late transition
                    curDiff[:t][curDiff[-t:] == 1] = 0 # early transition
                

                updatedDiff[startIdx:endIdx + 1] = curDiff

                # Cálculo de Jaccard, Precisión y Recall por fase
        for iPhase in range(1, nPhases + 1):
            mask_gt = (gtLabelID == iPhase).astype(int)
            mask_pred = (predLabelID == iPhase).astype(int)

            # Regiones conectadas para la fase actual
            _, gt_objects = label(mask_gt)
            _, pred_objects = label(mask_pred)

            if gt_objects == 0:
                # Si no hay presencia de la fase en GT, se asigna NaN
                res.append(np.nan)
                prec.append(np.nan)
                rec.append(np.nan)
                continue

            union_idx = np.where((gtLabelID == iPhase) | (predLabelID == iPhase))[-1]
            tp = np.sum(updatedDiff[union_idx] == 0)  # true positives relajados

            # Jaccard Index
            jaccard = tp / len(union_idx) * 100
            res.append(jaccard)
            
            # Precision
            pred_sum = np.sum(predLabelID == iPhase)
            prec.append(tp * 100 / (pred_sum + 0.0000000001))

            # Recall
            gt_sum = np.sum(gtLabelID == iPhase)
            rec.append(tp * 100 / gt_sum)

        # Exactitud total
        acc = np.sum(updatedDiff == 0) / len(gtLabelID) * 100

        return acc, prec, rec, res


def relaxed_evaluation_M2CAI(dataset_preds, dataset_gts, names):

    results = {}
    video_names = list(dict.fromkeys(names))


    dataset_info = {'Video_id': names, 'Pred': dataset_preds, 'GT': dataset_gts}
    df = pd.DataFrame(dataset_info)
    video_results = []
    
    jaccard = np.zeros((14, 8))
    precision = np.zeros((14, 8))
    recall = np.zeros((14, 8))
    accuracy = np.zeros(14)

    # Iterate through each video to calculate metrics
    for idx, video in enumerate(video_names):
        filtered_df = df[df['Video_id'] == video]
        video_preds = filtered_df['Pred'].tolist()
        video_gts = filtered_df['GT'].tolist()

        if video_preds and video_gts:
            video_results.append(calculate_metrics(video_preds, video_gts, ["accuracy", "precision", "recall", 'jaccard', 'f1_score']))

            acc, pre, rec, jac = relaxed_evaluation_M2CAI_video(np.array(video_preds), np.array(video_gts))

            accuracy[idx] = acc
            precision[idx, :] = pre
            recall[idx, :] = rec
            jaccard[idx, :] = jac


        else:
            raise ValueError(f"No predictions or ground truths found for video {video} in dataset.")
        
    metrics_mean = {metric: np.mean([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'jaccard', 'f1_score']}
    metrics_std = {metric: np.std([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'jaccard', 'f1_score']}

    # Relaxed metrics adaptations
    # Limitar valores máximos a 100
    jaccard = np.clip(jaccard, None, 100)
    precision = np.clip(precision, None, 100)
    recall = np.clip(recall, None, 100)

    mean_jacc_per_phase = np.nanmean(jaccard, axis=0)
    mean_prec_per_phase = np.nanmean(precision, axis=0)
    mean_rec_per_phase = np.nanmean(recall, axis=0)

    mean_jaccard, std_jaccard = np.mean(mean_jacc_per_phase), np.std(mean_jacc_per_phase)
    mean_precision, std_precision = np.mean(mean_prec_per_phase), np.std(mean_prec_per_phase)
    mean_recall, std_recall = np.mean(mean_rec_per_phase), np.std(mean_rec_per_phase)
    mean_accuracy, std_accuracy = np.mean(accuracy), np.std(accuracy)

    metrics_mean['relaxed accuracy'] = mean_accuracy
    metrics_mean['relaxed precision'] = mean_precision
    metrics_mean['relaxed recall'] = mean_recall
    metrics_mean['relaxed jaccard'] = mean_jaccard

    metrics_std['relaxed accuracy'] = std_accuracy
    metrics_std['relaxed precision'] = std_precision
    metrics_std['relaxed recall'] = std_recall
    metrics_std['relaxed jaccard'] = std_jaccard


    results['M2CAI'] = {"mean": metrics_mean, "std": metrics_std}

    return results