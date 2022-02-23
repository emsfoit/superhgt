"""
This script is an example of how to simplify or complexity the Reddit dataset.
The data is meant to be fed into the SHGT framework, but can be used by many other tools, Spiderman (Graphs) but also in Sets and Trees.
That is, this is a global benchmarking effort.
So far, the simplification is restricted to selecting a subset of the events in the submission trace.
"""
import pandas as pd
import bz2
import json
import ast
import argparse as ap
import sys

def load_args(argv):
    parser = ap.ArgumentParser()
    parser.add_argument('--sample_size_main_file', type=int, default=50000,
                        help='number of samples')
    parser.add_argument('--input_dir', type=str, default='dataset/reddit/',
                        help='The address of raw data.')
    parser.add_argument('--posts_file_name', type=str, default='RS_2011-01.bz2',
                        help='The address of raw data.')
    parser.add_argument('--comments_file_name', type=str, default='RC_2011-01.bz2',
                        help='The address of raw data.')
    parser.add_argument('--data_type', type=str, default='json',
                        help='The address of raw data.')
    parser.add_argument('--output_dir', type=str, default='dataset/output/reddit/',
                        help='The address for storing the preprocessed data.')                      
    parser.add_argument('--compress',type=bool, default= False,
                        help='compression option')
    parser.add_argument('--sep', type=str, default=';',
                        help='Seperator')
    parser.add_argument('--save_result',type=bool, default= True,
                        help='compression option')
    args = parser.parse_args()

    return args 


def filter_posts(args):
    """take sample_size_main_file recoed of the posts
    Returns
    -------
    posts dataframe
    """
    file_path = args.input_dir + args.posts_file_name
    posts_df = filter_posts_json(args, file_path)
    if args.save_result:
        if args.compress:
            posts_df.to_csv(
                args.output_dir + args.posts_file_name,
                sep=args.sep,
                index=False,
                compression="bz2",
            )
        else:
            file_name = args.posts_file_name.split(".")[0] + ".csv"
            posts_df.to_csv(args.output_dir + file_name, sep=args.sep, index=False)
    return posts_df


def filter_posts_json(args, file_path):
    posts_data = []
    with bz2.open(file_path, "rt", encoding="utf-8") as bzinput:
        for i, line in enumerate(bzinput):
            if len(posts_data) >= args.sample_size_main_file:
                break
            try:
                """
                extracting the data from file ( which is JSON by default)
                if it does exist, otherwise take empty strings (default)
                """
                post = json.loads(line)
                post_id = post.get("name", "")
                post_title = post.get("title", "")
                post_created_utc = post.get("created_utc", "")
                author = post.get("author", "")
                subreddit_id = post.get("subreddit_id", "")
                subreddit = post.get("subreddit", "")
                # filtering out invalid entries
                if (
                    post_id == ""
                    or post_title == ""
                    or post_created_utc == "" 
                    or len(post_title.split()) < 2 or "[deleted]" in post_title
                    or author == "" or "[deleted]" in author
                    or subreddit_id == "" or "[deleted]" in subreddit_id
                    or subreddit == "" or "[deleted]" in subreddit
                ):
                    continue
                posts_data.append(
                    [
                        post_id,
                        post_title,
                        post_created_utc,
                        author,
                        subreddit_id,
                        subreddit,
                    ]
                )
            except:
                print("could not load line" + str(i))
                continue
    posts_df = pd.DataFrame(
        posts_data,
        columns=[
            "post_id",
            "post_title",
            "post_created_utc",
            "author",
            "subreddit_id",
            "subreddit"
        ],
    )
    posts_df = posts_df.replace({";": ""}, regex=True).dropna().drop_duplicates()
    posts_df = posts_df.replace({"[deleted]": pd.NaT}).dropna().drop_duplicates()
    return posts_df
    # Tried this but it failed at some points 
    df = pd.DataFrame()
    iter_csv = pd.read_json(file_path, encoding='utf-8', lines=True, chunksize=1000)
    for chunk in iter_csv:
        try:
            # chunk = chunk[chunk['num_comments']>5]
            filtered = chunk[['name', 'title', 'created_utc', 'author', 'subreddit_id', 'subreddit']]
            # filtered = filtered.replace({"[deleted]": pd.NaT}).dropna().drop_duplicates()
            if len(filtered) > 0: df = pd.concat([df, filtered])
            if len(df) >= 100000:
                df = df[:args.sample_size_main_file]
                break
        except e:
            print(e)
    df = df.rename(columns={
        'name': "post_id",
        'title': "post_title",
        'created_utc': "post_created_utc"
    })
    df = df.replace({";": ""}, regex=True)
    return df

def filter_comments(args, filtered_posts):
    """filter comments according to the filtered_posts
    Parameters
    ----------
    filtered_posts : dataframe
        posts are used in filtering
    Returns
    -------
    dataframe
        comments dataframe
    """
    file_path = args.input_dir + args.comments_file_name
    comments_df = filter_comments_json(file_path, filtered_posts)

    if args.save_result:
        if args.compress:
            comments_df.to_csv(
                args.output_dir + args.comments_file_name,
                sep=";",
                index=False,
                compression="bz2",
            )
        else:
            file_name = args.comments_file_name.split(".")[0] + ".csv"
            comments_df.to_csv(
                args.output_dir + file_name, sep=args.sep, index=False
            )
    return comments_df

def filter_comments_json(file_path, filtered_posts):
    post_ids = filtered_posts["post_id"].to_list()
    df = pd.DataFrame()
    iter_csv = pd.read_json(file_path, encoding='utf-8', lines=True, chunksize=50000)
    for chunk in iter_csv:
        filtered = chunk[['name', 'body', 'created_utc', 'parent_id', 'author']]
        comment_ids_with_post = (
            filtered[filtered["parent_id"].isin(post_ids)]["name"]
            .to_list()
        )
        available_parent_ids = post_ids + comment_ids_with_post
        comments_df = filtered[filtered["parent_id"].isin(available_parent_ids)]
        comments_df["post_id"] = comments_df[comments_df["parent_id"].str.contains("t3_")]["parent_id"]
        comments_df["comment_parent_id"] = comments_df[comments_df["parent_id"].str.contains("t1_")]["parent_id"]
        comments_df = comments_df.drop(columns=["parent_id"])
        comments_df = comments_df.replace({";": ""}, regex=True)
        df = pd.concat([df, comments_df])
    df = df.rename(columns={
            'name': "comment_id",
            'body': "comment_body",
            'created_utc': "comment_created_utc"
    })
    df = df.replace({"[deleted]": pd.NaT}).dropna(subset=['comment_id', 'comment_body', 'author'])

    return df



def main(argv):
    args = load_args(argv)
    posts = filter_posts(args)
    comments = filter_comments(args, posts)

    return posts, comments

if __name__ == '__main__':
    main(sys.argv)