import csv
import argparse
from datetime import datetime


def process_facebook_posts(fb_file, output_file):
    rows = []
    with open(fb_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            message = row['message'] or ''
            created = row['created_time']
            try:
                post_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
            except Exception:
                post_date = None
            rows.append({
                'message': message,
                'post_date': post_date.strftime('%Y-%m-%d') if post_date else '',
                'post_impressions': row.get('post_impressions', '')
            })
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['message', 'post_date', 'post_impressions'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Prepare facebook data for tactical features.')
    parser.add_argument('--fb-file', default='merged_facebook_posts_insights - merged_facebook_posts_insights.csv')
    parser.add_argument('--output-file', default='facebook_posts_simple.csv')
    args = parser.parse_args()
    process_facebook_posts(args.fb_file, args.output_file)


if __name__ == '__main__':
    main()
