__author__ = '{Alfonso Aguado Bustillo}'

# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np


# Create a function
def select_n_components(var_ratio, goal_var):

    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


def feature_extraction(data):

    features = data.loc[:, ['title_length', 'n_title_words', 'n_introduction_words', 'n_full_text_words',
                 'bytes_introduction_text', 'bytes_full_text', 'n_citations', 'n_sections',
                 'n_internal_links', 'n_external_links', 'n_image_links', 'n_back_links', 'n_lang_links',
                 'n_contributors', 'n_contributors_edits', 'time_since_last_edit', 'time_since_first_edit',
                 'n_categories',
                 'is_redirect', 'is_category_page', 'is_category_redirect', 'is_disambig', 'is_talkpage', 'is_filepage'
                            ]].values

    # Standardize feature matrix
    features = StandardScaler().fit_transform(features.data)
    # Make sparse matrix
    features_sparse = csr_matrix(features)

    # # Show results
    print("Original number of features:", features_sparse.shape[1])
    # print("Reduced number of features:", features_sparse_tsvd.shape[1])

    # Create and run an TSVD with one less than number of features
    tsvd = TruncatedSVD(n_components=features_sparse.shape[1] - 1)
    features_tsvd = tsvd.fit(features)
    # List of explained variances
    tsvd_var_ratios = tsvd.explained_variance_ratio_

    # Run function
    n_components = select_n_components(tsvd_var_ratios, 0.95)
    print("Reduced number of features:", n_components)

    # Create a TSVD
    tsvd = TruncatedSVD(n_components=n_components)
    # Conduct TSVD on sparse matrix
    features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)