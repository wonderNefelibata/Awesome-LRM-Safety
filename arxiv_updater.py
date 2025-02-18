import arxiv
import datetime
import json
from tenacity import retry, stop_after_attempt, wait_fixed

# 查询关键词配置
query_terms = {
    'include': [
        "large reasoning model safety",
        "AI safety",
        "language model security",
        "adversarial attacks",
        "machine learning security"
    ],
    'exclude': []
}

# @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))  # 最多重试3次，每次间隔5秒
def fetch_papers():
    # 构建查询条件
    query_parts = []
    for term in query_terms['include']:
        query_parts.append(f'all:"{term}"')
    for term in query_terms['exclude']:
        query_parts.append(f'-all:"{term}"')
    query = ' OR '.join(query_parts)
    
    # 查询 arXiv
    search = arxiv.Search(
        query=query,
        max_results=50,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # 使用 Client.results 获取结果
    client = arxiv.Client delay_seconds=3)  # 设置请求间隔
    results = client.results(search)
    
    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "url": result.entry_id,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "summary": result.summary.replace('\n', ' ')[:150] + '...'
        })
    return papers

def update_article_json(papers):
    # 保存所有论文到 article.json
    try:
        with open('article.json', 'r') as f:
            existing_papers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_papers = []  # 文件不存在或为空时，初始化为空列表

    # 去重并更新
    unique_papers = []
    paper_urls = [p['url'] for p in existing_papers]
    for paper in papers:
        if paper['url'] not in paper_urls:
            unique_papers.append(paper)
    
    updated_papers = existing_papers + unique_papers
    with open('article.json', 'w') as f:
        json.dump(updated_papers, f, indent=2)

def update_markdown():
    # 读取 JSON 数据
    try:
        with open('article.json', 'r') as f:
            all_papers = json.load(f)
    except FileNotFoundError:
        all_papers = []
    
    # 按时间降序排序
    all_papers_sorted = sorted(all_papers, key=lambda x: x['published'], reverse=True)
    latest_papers = all_papers_sorted[:30]  # 显示最新30篇
    older_papers = all_papers_sorted[30:]    # 历史论文
    
    # 生成 Markdown 表格
    def generate_table(papers, title):
        if not papers:
            return ""
        table = f"\n\n--- \n\n## {title}\n\n| Date       | Title                                      | Authors           | Abstract                                      |\n|------------|--------------------------------------------|-------------------|-----------------------------------------------|\n"
        for p in papers:
            table += f"| {p['published'][:10]} | [{p['title']}]({p['url']}) | {', '.join(p['authors'][:2])} et al. | {p['summary']} |\n"
        table += "\n"
        return table
    
    # 更新 README.md
    with open('README.md', 'r') as f:
        content = f.read()
    
    # 插入最新论文（未折叠）
    new_content = content.replace('<!-- ARXIV_PAPERS_START -->', generate_table(latest_papers, "Latest arXiv Papers (Auto-Updated)"))
    
    # 插入历史论文（折叠样式）
    history_table = '\n\n'.join([f"<details><summary>View Older Papers</summary>{generate_table(older_papers, 'Historical arXiv Papers')}</details>" if older_papers else ""])
    new_content = new_content.replace('<!-- ARXIV_PAPERS_HISTORY -->', history_table)
    
    with open('README.md', 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    papers = fetch_papers()
    update_article_json(papers)
    update_markdown()
