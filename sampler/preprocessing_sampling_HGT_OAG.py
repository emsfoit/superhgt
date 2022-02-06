"""This script can be used to structure (and take a sample of) OAG data in a way that can be used by our model
    input:
    ------
    OAG data
    https://drive.google.com/drive/folders/1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz

    output:
    -------
    The following .tsv files :
        - papers_papers: ['paper_id', 'paper_parent_id']
        - papers_authors_affiliations: ['paper_id', 'publish_year', 'author_id', 'affiliation_id','author_position']
        - papers_fields: ['paper_id', 'publish_year', 'field_id', 'level']
        - fields_hierarchy: ['field_id', 'field_parent_id']
        - papers_venues: ['paper_id', 'publish_year', 'paper_title', 'venue_id', 'venue_type']
        - fields: ['field_id', 'field_name', 'field_emb']
        - affiliations: ['affiliation_id', 'affiliation_name', 'affiliation_emb']
        - venues: ['venue_id', 'venue_name', 'venue_emb']
"""
import ast
import argparse as ap
import pandas as pd
import numpy as np
import sys
import gc
import time


def load_args(argv=None):
    """a function to load and parse the arguments and pack them into
        a dict to be globally used.

    Args:
        argv ([type], optional): arguments passed from main(). Defaults to None.

    Returns:
        dict: data which contains all the IO data
    """
    parser = ap.ArgumentParser()
    parser.add_argument('--sample_size_main_file', type=int, default=50000,
                        help='number of samples')
    parser.add_argument('--input_dir', type=str, default='dataset/raw_data/',
                        help='The address of raw data.')
    parser.add_argument('--output_dir', type=str, default='dataset/output/',
                        help='The address for storing the preprocessed data.')
    parser.add_argument('--time', type=ast.literal_eval, default=[{'min_time': 2000, 'max_time': 2019}],
                        help='time range')
    parser.add_argument('--compress', type=bool, default=False,
                        help='compression option')

    args = parser.parse_args()
    # make a dict so we add papers and co for further dependency.
    sample_size = argv if argv and isinstance(argv, int) and argv != 0 else args.sample_size_main_file
    data = {'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'sample_size_main_file': sample_size,
            'time': args.time,
            'compress': args.compress,
            }
    return data


def save_data(df, filename, data, header=True):
    dataframe = df.replace('', np.nan).dropna(axis='rows', how='any')
    if data['compress']:
        dataframe.to_csv(data['output_dir'] + filename + '.gz',
                         index=False, sep='\t', compression='gzip', header=header)
    else:
        dataframe.to_csv(data['output_dir'] + filename,
                         index=False, sep='\t', header=header)


def load_filter_by_time(df, sample_size_main_file=1000, time_key=None, time=[]):
    """ a function to filter by time, taking a list of time dicts (objects) and
        returns entries (which count is specified by sample_size_main_file param) for each time
        object in the list.

    Args:
        df: dataframe to filter
        sample_size_main_file (int, optional): count of entries(samples). Defaults to 10000.
        time_key (str, optional): the name of time attribute. Defaults to None.
        time (list, optional): list of time dicts (objects). Defaults to [].

    Returns:
        DataFrame: filtered entries from specified file by specifed time key
    """
    columns = df.columns.to_list()
    df[time_key] = pd.to_numeric(df[time_key])
    result = pd.DataFrame(columns=columns)
    for elm in time:
        min_time = elm['min_time'] if 'min_time' in elm else float('-inf')
        max_time = elm['max_time'] if 'max_time' in elm else float('inf')
        filtered = df[(
            df[time_key] >= min_time)
            & (df[time_key] <= max_time)
            & (df['lang'] == 'en')
            & (df['normalized_title'].str.len() >= 4)][:sample_size_main_file]
        result = pd.concat([result, filtered])
    return result


def author_position_df(df):
    if str(df['AuthorSequenceNumber']) == '1':
        return 'write_first'
    elif str(df['AuthorSequenceNumber']) == str(df['max_authors']):
        return 'write_last'
    else:
        return 'write_other'


def preprocessing_main_file(df):
    df = df[['PaperId', 'PublishYear', 'NormalizedTitle', 'VenueId',
             'DetectedLanguage', 'DocType']]
    df.columns = ['paper_id', 'publish_year',
                  'normalized_title', 'venue_id', 'lang', 'venue_type']
    return df


def preprocessing_fields_files(fields_h, paper_fields):
    """
        Variables:
        fields_012345 : all fields from L0 to L5
        fields_12345 : all fields from L1 to L5 (children or (parents with own parents))
        fields_0 : all fields with level L0 (roots with no parents)
        fields_hierarchy
    """
    fields_012345 = fields_h
    fields_12345 = fields_012345[['ChildFosId',
                                  'ParentFosId', 'ChildLevel']].drop_duplicates()
    fields_0 = fields_012345[fields_012345['ParentLevel']
                             == 'L0'][['ParentFosId', 'ParentLevel']]
    del fields_012345
    fields_12345.columns = ['field_id', 'field_parent_id', 'level']
    fields_0.columns = ['field_id', 'level']
    fields_hierarchy = fields_12345.append(fields_0).drop_duplicates()
    del fields_12345, fields_0
    paper_field = paper_fields
    paper_field.columns = ['paper_id', 'field_id']
    paper_field = paper_field.merge(
        fields_hierarchy[['field_id', 'level']].drop_duplicates(), how='left', on='field_id')
    fields_hierarchy = fields_hierarchy[['field_id', 'field_parent_id']]
    return paper_field, fields_hierarchy


def preprocessing_pr_file(df):
    # ['PaperId', 'ReferenceId']
    df.columns = ['paper_id', 'paper_parent_id']
    return df


def preprocessing_PAuAf_file(df):
    # ['PaperSeqid', 'AuthorSeqid', 'AffiliationSeqid', 'AuthorSequenceNumber']
    temp = df.groupby(['PaperSeqid']).agg(
        {'AuthorSequenceNumber': 'max'}).reset_index().rename(columns={'AuthorSequenceNumber': 'max_authors'})
    paper_author_affiliation = df.merge(
        temp, how='left', on='PaperSeqid')
    del temp
    paper_author_affiliation['author_position'] = paper_author_affiliation.apply(
        author_position_df, axis=1)
    paper_author_affiliation = paper_author_affiliation[[
        'PaperSeqid', 'publish_year', 'AuthorSeqid', 'AffiliationSeqid', 'author_position']]
    paper_author_affiliation.columns = [
        'paper_id', 'publish_year', 'author_id', 'affiliation_id', 'author_position']
    return paper_author_affiliation


def preprocessing_seq_and_vfi_files(seq, vfi_vector):
    # affiliations (institutes):
    affiliations = seq[(seq['type'] == 'affiliation')].merge(
        vfi_vector, how='left', left_on='id', right_on='vfi_id')[['id', 'name', 'emb']]
    affiliations.columns = ['affiliation_id',
                            'affiliation_name', 'affiliation_emb']
    # venues:
    venues = seq[seq['type'].isin(['journal', 'conference'])].merge(
        vfi_vector, how='left', left_on='id', right_on='vfi_id')[['id', 'name', 'emb']]
    venues.columns = ['venue_id', 'venue_name', 'venue_emb']
    # fields:
    fields = seq[(seq['type'] == 'fos')]
    fields = fields.merge(
        vfi_vector, how='left', left_on='id', right_on='vfi_id')[['id', 'name', 'emb']]
    fields.columns = ['field_id', 'field_name', 'field_emb']
    return affiliations, venues, fields


def main(argv):
    sampling_start_time = time.time()
    print("started sampling..")
    # loading args
    data = load_args(argv)

    start_time = time.time()
    print("############################")
    print("started processing papers..")
    papers = pd.read_csv(data['input_dir'] +
                         'Papers_CS_20190919.tsv', sep="\t", dtype=str)
    papers = preprocessing_main_file(papers)
    papers = load_filter_by_time(
        papers, time_key='publish_year', time=data['time'],
        sample_size_main_file=data['sample_size_main_file'])[
        ['paper_id', 'publish_year', 'normalized_title', 'venue_id', 'venue_type']]
    save_data(papers, 'papers_venues.tsv', data)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done processing papers after: ", time_elapsed, "seconds")

    start_time = time.time()
    print("############################")
    print("started processing references..")
    paper_reference = pd.read_csv(
        data['input_dir'] + 'PR_CS_20190919.tsv', sep='\t', dtype=str)
    paper_reference = preprocessing_pr_file(paper_reference)
    paper_reference = paper_reference[paper_reference['paper_id'].isin(
        papers['paper_id'])]
    save_data(paper_reference, 'papers_papers.tsv', data)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done processing references after: ", time_elapsed, "seconds")
    del paper_reference

    start_time = time.time()
    print("############################")
    print("started processing fields..")
    fields_012345 = pd.read_csv(
        data['input_dir'] + 'FHierarchy_20190919.tsv', sep='\t', dtype=str)
    paper_fields = pd.read_csv(
        data['input_dir'] + 'PF_CS_20190919.tsv', sep='\t', dtype=str)[['PaperId', 'FieldOfStudyId']]
    paper_fields, fields_h = preprocessing_fields_files(
        fields_012345, paper_fields)
    del fields_012345
    paper_fields = paper_fields.merge(papers, on='paper_id')[
        ['paper_id', 'publish_year', 'field_id', 'level']]
    fields_h = fields_h[(fields_h['field_id'].isin(paper_fields['field_id'])) & (
        fields_h['field_parent_id'].isin(paper_fields['field_id']))]
    save_data(paper_fields, 'papers_fields.tsv', data)
    save_data(fields_h, 'fields_hierarchy.tsv', data)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done processing fields after: ", time_elapsed, "seconds")
    del fields_h

    start_time = time.time()
    print("############################")
    print("started processing PAuAf..")
    paper_author_affiliation = pd.read_csv(
        data['input_dir'] + 'PAuAf_CS_20190919.tsv', sep='\t', dtype=str)
    paper_author_affiliation = paper_author_affiliation.merge(
        papers, right_on='paper_id', left_on='PaperSeqid')
    paper_author_affiliation = preprocessing_PAuAf_file(
        paper_author_affiliation)
    save_data(paper_author_affiliation,
              'papers_authors_affiliations.tsv', data)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done processing PAuAf after: ", time_elapsed, "seconds")

    start_time = time.time()
    print("############################")
    print("started extracting fields, venues and affiliations..")
    seq_columns = ['id', 'name', 'type']
    seq = pd.read_csv(data['input_dir'] + 'SeqName_CS_20190919.tsv',
                      sep='\t', names=seq_columns, dtype=str)
    vfi_columns = ['vfi_id', 'emb']
    vfi_vector = pd.read_csv(data['input_dir'] + 'vfi_vector.tsv',
                             sep='\t', names=vfi_columns, header=None, dtype=str)
    affiliations, venues, fields = preprocessing_seq_and_vfi_files(
        seq, vfi_vector)
    affiliations = affiliations[affiliations['affiliation_id'].isin(
        paper_author_affiliation['affiliation_id'])]
    venues = venues[venues['venue_id'].isin(papers['venue_id'])]
    fields = fields[fields['field_id'].isin(paper_fields['field_id'])]
    del seq, vfi_vector, paper_author_affiliation
    save_data(affiliations, 'affiliations.tsv', data)
    save_data(venues, 'venues.tsv', data)
    save_data(fields, 'fields.tsv', data)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("done extracting after: ", time_elapsed, "seconds")
    del affiliations, venues, fields

    sampling_end_time = time.time()
    total_sampling_time = (sampling_end_time - sampling_start_time)
    print("done sampling after: ", total_sampling_time, "seconds")
    gc.collect()


if __name__ == '__main__':
    main(sys.argv)
