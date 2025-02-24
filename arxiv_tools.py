import arxiv
import json
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_papers_by_title(title_keywords):
    """通过标题精确查找 arXiv 论文"""
    try:
        query=f'all:"ti:{title_keywords}"'
        print(query)
        # 构造标题搜索查询
        search = arxiv.Search(
            query=f'all:"ti:{title_keywords}"',  # 使用标题精确查询
            max_results=1, 
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        client = arxiv.Client(delay_seconds=1)
        results = client.results(search)

        # 提取论文信息
        papers = []
        for result in results:
            paper = {
                "title": result.title,
                "url": result.entry_id,
                "arxiv_id": result.get_short_id(),
                "authors": [a.name for a in result.authors],
                "published": result.published.isoformat(),
                "summary": result.summary.replace('\n', ' ')  # 移除换行符，保留完整摘要
            }
            papers.append(paper)

        return papers

    except Exception as e:
        print(f"Error fetching papers by title: {e}")
        return []

def write_to_article_json(papers):
    """将论文信息写入 article.json 文件"""
    try:
        # 加载现有数据
        try:
            with open('article.json', 'r') as f:
                existing_papers = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_papers = []

        # 更新论文列表，合并新旧数据
        paper_ids = {paper['arxiv_id'] for paper in existing_papers}
        new_papers = []
        for paper in papers:
            if paper['arxiv_id'] not in paper_ids:
                new_papers.append(paper)
                paper_ids.add(paper['arxiv_id'])

        # 合并并保存
        merged_papers = existing_papers + new_papers
        with open('article.json', 'w') as f:
            json.dump(merged_papers, f, indent=2)

        print(f"Successfully added {len(new_papers)} new papers to article.json")

    except Exception as e:
        print(f"Error writing to article.json: {e}")

if __name__ == "__main__":
    # 示例：通过标题查找论文
    title_to_find = "o3-mini vs DeepSeek-R1: Which One is Safer?"
    papers = fetch_papers_by_title(title_to_find)
    # write_to_article_json(papers)
    print(papers)