"""
Auto-update script for arXiv papers
Query keywords can be modified in the 'query' parameter below
"""

import arxiv
import datetime

def fetch_papers():
    # 修改以下查询关键词以调整搜索范围
    search = arxiv.Search(
        query=(
            'all:"large reasoning model safety" OR '
            'all:"AI safety" OR '
            'all:"language model security" OR '
            'all:"adversarial attacks" OR '
            'all:"machine learning security"'
        ),
        max_results=15,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "url": result.entry_id,
            "authors": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary.replace('\n', ' ')[:150] + '...'  # 摘要截断
        })
    return papers

def update_readme(papers):
    with open('README.md', 'r') as f:
        content = f.read()
    
    table = "| Date | Title | Authors | Abstract |\n"
    table += "|------|-------|---------|----------|\n"
    for paper in papers[:10]:  # 显示最新10篇
        table += f"| {paper['published']} | [{paper['title']}]({paper['url']}) | {', '.join(paper['authors'][:2])} et al. | {paper['summary']} |\n"
    
    new_content = content.split('<!-- ARXIV_PAPERS_START -->')[0] + \
        f"<!-- ARXIV_PAPERS_START -->\n{table}\n<!-- ARXIV_PAPERS_END -->" + \
        content.split('<!-- ARXIV_PAPERS_END -->')[1]
    
    with open('README.md', 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    papers = fetch_papers()
    update_readme(papers)
