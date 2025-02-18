import arxiv
import datetime
import json
import re
from tenacity import retry, stop_after_attempt, wait_fixed

# 配置参数
QUERY_TERMS = {
    'include': [
        "DeepSeek-R1",
        "DeepSeek R1",
        "o1",
        "o3",
        "o3-mini",
        "safety",
        "jailbreak",
        "Long Chain-of-Thought Reasoning",
        "CoT-enabled models",
    ],
    'exclude': []
}
MAX_NEW_PAPERS = 100
LATEST_PAPERS_COUNT = 30

def extract_arxiv_id(url):
    """从arXiv URL中提取基础ID（不含版本号）"""
    match = re.search(r'/(?:abs|pdf)/(.*?)(?:v\d+)?(?:\.pdf)?$', url)
    return match.group(1) if match else None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_papers():
    """从arXiv获取最新论文"""
    query = " OR ".join([f'all:"{term}"' for term in QUERY_TERMS['include']])
    search = arxiv.Search(
        query=query,
        max_results=MAX_NEW_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    client = arxiv.Client(delay_seconds=1)
    return [{
        "title": result.title,
        "url": result.entry_id,
        "arxiv_id": result.get_short_id(),
        "authors": [a.name for a in result.authors],
        "published": result.published.isoformat(),
        "summary": result.summary.replace('\n', ' ')[:150] + '...'
    } for result in client.results(search)]

def update_article_json(new_papers):
    """更新论文数据库"""
    try:
        with open('article.json', 'r') as f:
            existing_papers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_papers = []

    # 合并新旧论文并去重
    papers_dict = {}
    for paper in existing_papers + new_papers:
        # 处理旧数据缺少arxiv_id的情况
        if 'arxiv_id' not in paper:
            paper['arxiv_id'] = extract_arxiv_id(paper['url'])
        
        # 保留最新版本
        pid = paper['arxiv_id']
        existing = papers_dict.get(pid)
        if not existing or datetime.datetime.fromisoformat(paper['published']) > datetime.datetime.fromisoformat(existing['published']):
            papers_dict[pid] = paper

    updated_papers = sorted(papers_dict.values(), 
                          key=lambda x: x['published'], 
                          reverse=True)
    
    with open('article.json', 'w') as f:
        json.dump(updated_papers, f, indent=2)
    print(f"论文数据库已更新，当前总数：{len(updated_papers)}篇")

def generate_markdown_table(papers, title=""):
    """生成Markdown表格"""
    if not papers:
        return ""
    
    table = ""
    if title:
        table += f"\n\n## {title}\n\n"
    table += "| Date       | Title                                      | Authors           | Abstract Summary          |\n"
    table += "|------------|--------------------------------------------|-------------------|---------------------------|\n"
    
    for p in papers:
        authors = ', '.join(p['authors'][:2]) + (' et al.' if len(p['authors']) > 2 else '')
        table += f"| {p['published'][:10]} | [{p['title']}]({p['url']}) | {authors} | {p['summary']} |\n"
    return table

def update_readme():
    """更新README文件"""
    # 读取论文数据
    try:
        with open('article.json', 'r') as f:
            all_papers = sorted(json.load(f), 
                              key=lambda x: x['published'], 
                              reverse=True)
    except FileNotFoundError:
        all_papers = []
    
    # 分割最新和历史论文
    latest = all_papers[:LATEST_PAPERS_COUNT]
    historical = all_papers[LATEST_PAPERS_COUNT:]
    
    # 生成最新表格
    latest_table = generate_markdown_table(latest, "Latest arXiv Papers (Auto-Updated)")
    
    # 生成历史表格（可折叠）
    history_section = ""
    if historical:
        history_table = generate_markdown_table(historical, "Historical Papers")
        history_section = f"""
<details>
<summary>📚 View Historical Papers ({len(historical)} entries)</summary>

{history_table}
</details>
"""
    # 更新README内容
    with open('README.md', 'r+', encoding='utf-8') as f:
        content = f.read()
        placeholder = '<!-- ARXIV_PAPERS -->'
        start = content.find(placeholder)
        
        if start != -1:
            new_content = content[:start + len(placeholder)] + latest_table + history_section
            f.seek(0)
            f.truncate()
            f.write(new_content)
            print("README更新成功！")
        else:
            print("⚠️ 未找到占位符，请确认README中包含<!-- ARXIV_PAPERS -->")

if __name__ == "__main__":
    new_papers = fetch_papers()
    update_article_json(new_papers)
    update_readme()
