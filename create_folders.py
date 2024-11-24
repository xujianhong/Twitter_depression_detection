import os

def createFoldersResults(directory, lang='eng', flagFolds=True, folds=5):
    """
	directory: Directory where the folders will be created 
	lang: 3 letters for language (esp - español, eng - english...)
	flagFolds: State whether folders are to be created to store the results of each fold (True/False)
	folds: number of folds (5, 10...)
	"""
    # 10k --> vocabulary size is 10,000 n-grams
	# lda50 --> 50 is the number of topics used
	# 300d --> 300 is the size of the embeddings used
    # lst1 = ['word_unigram_10k','word_bigram_10k','word_trigram_10k','word_unibitrigram_10k']
    # lst2 = ['postag_unigram_10k','postag_bigram_10k','postag_trigram_10k','postag_unibitrigram_10k']
    # lst3 = ['character-wb_3gram_10k','character-wb_5gram_10k','character-wb_7gram_10k']
    # lst4 = ['topic_lda50_10k','topic_nmf50_10k']
    # lst5 = ['style_lf_liwc','style_lf_all']
    # lst6 = ['embeddings_glove_300d','embedding_fasttest_300d']

    # lst = lst1+lst2+lst3+lst4+lst5+lst6

    # directoriesNames =[f'{e}_{lang}' for e in lst]

    # for nameDir in directoriesNames:
    #     pathDir = os.path.join(directory, nameDir.upper())
    #     os.mkdir(pathDir)
    #     print(f'\nDirectory {nameDir} created')

        # if flagFolds:
        #     for i in range(1, folds+1):
        #         pathFold = os.path.join(pathDir, f'Fold{i}')
        #         os.mkdir(pathFold)
        #         print(f'> {nameDir}\Fold{i}')
    if flagFolds:
            for i in range(1, folds+1):
                pathFold = os.path.join(directory, f'Fold{i}')
                os.mkdir(pathFold)
                print(f'> {directory}\Fold{i}')
    print('\n\nDirectories created! :)')


resultsDir = r'/Users/jianhongxu/python_project/twitter_dataset/Results'
createFoldersResults(resultsDir)