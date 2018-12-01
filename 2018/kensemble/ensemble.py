import pandas as pd
import os
import numpy as np
import scipy.stats

def ensembleVer2(input_folder, output_path):
    print('Out:' + output_path)
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    model_scores = []
    for i, csv in enumerate(csv_files):
        df = pd.read_csv(os.path.join(input_folder, csv), index_col=0)
        if i == 0:
            index = df.index
        else:
            assert index.equals(df.index), "Indices of one or more files do not match!"
        model_scores.append(df)
    print("Read %d files. Averaging..." % len(model_scores))

    # print(model_scores)
    concat_scores = pd.concat(model_scores)
    print(concat_scores.head())
    # concat_scores['is_iceberg'] = concat_scores['is_iceberg'].astype(np.float32)

    # averaged_scores = concat_scores.groupby(['file','species']).agg(lambda x:x.value_counts().index[0])

    averaged_scores=concat_scores.groupby(['file']).agg(lambda x: scipy.stats.mode(x)[0][0])

    # averaged_scores=concat_scores.groupby(['file', 'species']).agg(lambda x: scipy.stats.mode(x)[0][0])

    assert averaged_scores.shape[0] == len(list(index)), "Something went wrong when concatenating/averaging!"
    # averaged_scores = averaged_scores.reindex(index)

    # stacked_1 = pd.read_csv('sample_submission.csv')  # for the header
    # print(stacked_1.shape)
    # sub = pd.DataFrame()
    # sub['file'] = stacked_1['file']
    #
    # sub['species'] = np.exp(np.mean(
    #     [
    #         averaged_scores['species'].apply(lambda x: np.log(x))
    #     ], axis=0))
    #
    # print(sub.shape)
    #
    averaged_scores.to_csv(output_path, index=True, float_format='%.9f')
    print("Averaged scores saved to %s" % output_path)

if __name__ == '__main__':

    ensembleVer2('ensemble_csvs','ens.csv')
