import xhs
from xhs import FeedType, IPBlockError, XhsClient
from xhs.exception import SignError, DataFetchError
import json
from time import sleep
from playwright.sync_api import sync_playwright

def beauty_print(data: dict):
    print(json.dumps(data, ensure_ascii=False, indent=2))

def test_create_simple_note(xhs_client: XhsClient, image_paths: list, title: str, desc: str):
    # title = "新闻测试页"
    # desc = "我是新闻我是新闻我是新闻"
    note = xhs_client.create_image_note(title, desc, image_paths, is_private=False, post_time="2023-07-25 23:59:59")
    beauty_print(note)

def sign(uri, data=None, a1="", web_session=""):
    for _ in range(10):
        try:
            with sync_playwright() as playwright:
                stealth_js_path = "./stealth.min.js"
                chromium = playwright.chromium

                # 如果一直失败可尝试设置成 False 让其打开浏览器，适当添加 sleep 可查看浏览器状态
                browser = chromium.launch(headless=True)

                browser_context = browser.new_context()
                browser_context.add_init_script(path=stealth_js_path)
                context_page = browser_context.new_page()
                context_page.goto("https://www.xiaohongshu.com")
                browser_context.add_cookies([
                    {'name': 'a1', 'value': a1, 'domain': ".xiaohongshu.com", 'path': "/"}]
                )
                context_page.reload()
                # 这个地方设置完浏览器 cookie 之后，如果这儿不 sleep 一下签名获取就失败了，如果经常失败请设置长一点试试
                sleep(1)
                encrypt_params = context_page.evaluate("([url, data]) => window._webmsxyw(url, data)", [uri, data])
                return {
                    "x-s": encrypt_params["X-s"],
                    "x-t": str(encrypt_params["X-t"])
                }
        except Exception as e:
            print(e)
            # 这儿有时会出现 window._webmsxyw is not a function 或未知跳转错误，因此加一个失败重试趴
            pass
    raise Exception("重试了这么多次还是无法签名成功，寄寄寄")