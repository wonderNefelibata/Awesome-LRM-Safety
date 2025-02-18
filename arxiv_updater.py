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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))  # 最多重试3次，每次间隔5秒
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
    client = arxiv.Client(delay_seconds=1)  # 设置请求间隔
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
    print("papers: ", len(papers))
    # 保存所有论文到 article.json
    try:
        with open('article.json', 'r') as f:
            print("article.json exists, updating...")
            existing_papers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("article.json not found, creating new one.")
        existing_papers = []  # 文件不存在或为空时，初始化为空列表

    print("existing_papers的长度: ", len(existing_papers))
    # 去重并更新
    unique_papers = []
    paper_urls = [p['url'] for p in existing_papers]
    for paper in papers:
        if paper['url'] not in paper_urls:
            unique_papers.append(paper)

    print("unique_papers的长度: ", len(unique_papers))
    
    updated_papers = existing_papers + unique_papers
    print("updated_papers的长度: ", len(updated_papers))
    with open('article.json', 'w') as f:
        print("article.json存在，准备更新")
        json.dump(updated_papers, f, indent=2)
        print(f"成功更新 {len(unique_papers)} 篇新论文")

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
    def generate_table(papers):
        if not papers:
            return ""
        table = f"\n\n| Date       | Title                                      | Authors           | Abstract                                      |\n|------------|--------------------------------------------|-------------------|-----------------------------------------------|\n"
        for p in papers:
            table += f"| {p['published'][:10]} | [{p['title']}]({p['url']}) | {', '.join(p['authors'][:2])} et al. | {p['summary']} |\n"
        table += "\n"
        return table
    
    # 更新 README.md
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找占位符位置
    placeholder = '<!-- ARXIV_PAPERS_START -->'
    placeholder_pos = content.find(placeholder)
    if placeholder_pos == -1:
        print("Warning: 占位符 '" + placeholder + "' 未找到，无法更新论文内容！")
        return
    
    # 提取占位符前面的内容和后面的内容
    prefix = content[:placeholder_pos + len(placeholder)]
    suffix = content[placeholder_pos + len(placeholder):]
    
    # 生成最新论文表格
    latest_table = generate_table(latest_papers)
    # 生成历史论文表格
    history_table = '\n\n'.join([f"<details><summary>View Older Papers</summary>{generate_table(older_papers)}</details>" if older_papers else ""])
    
    # 组装新的内容
    new_content = prefix + latest_table + history_table + suffix
    
    # 写入新内容
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == "__main__":
    papers = fetch_papers()
    update_article_json(papers)
    update_markdown()
