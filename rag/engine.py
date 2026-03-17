"""
RAG引擎：文档加载、切分、存储、检索
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os, pathlib

class RagEngine:
    def __init__(self, persist_dir: str = None):
        self.script_dir = pathlib.Path(__file__).parent
        self.persist_dir = persist_dir or str(self.script_dir / "chroma_db")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        self.vectorstore = None
        self._load_or_create()

    def _load_or_create(self):
        """加载已有的向量库，没有就创建"""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            print(f"RAG引擎：加载已有向量库，共{self.vectorstore._collection.count()}个文本块")
        else:
            # 首次启动，加载默认知识库
            self._init_default_knowledge()

    def _init_default_knowledge(self):
        """加载默认知识库文档"""
        docs_dir = self.script_dir / "docs"
        if not docs_dir.exists():
            docs_dir.mkdir(parents=True)
            # 创建默认产品手册
            self._create_default_docs(docs_dir)

        self.load_directory(str(docs_dir))

    def _create_default_docs(self, docs_dir):
        """创建默认产品手册"""
        content = """# 智选商城产品手册

## 退换货政策
所有商品支持7天无理由退换货，需保持商品及包装完好。生鲜食品、定制商品除外。退货运费由买家承担，换货运费由平台承担。退款将在收到退货后3个工作日内原路退回。

## 会员权益
普通会员：每月3次免运费机会，生日当月享9折优惠。
黄金会员：每月10次免运费，全场95折，优先客服通道。升级条件：年消费满2000元。
钻石会员：无限免运费，全场9折，专属客服1对1服务，每季度赠送50元优惠券。升级条件：年消费满5000元。

## 手机壳系列
材质：采用航空级TPU材质，耐摔防刮。支持iPhone 15/16全系列和华为Mate 60/70系列。
价格：普通款39元，磁吸款69元，联名限定款99-199元。
保修：非人为损坏30天内免费更换。

## 充电器系列
Type-C快充充电器：支持65W快充，兼容苹果、安卓、笔记本。售价89元。
无线充电板：支持15W无线快充，兼容Qi协议设备。售价129元。
保修期：均为1年，非人为损坏免费更换。

## 配送说明
默认顺丰快递，下单后48小时内发货。偏远地区可能延迟1-2天。
支持到店自取，全国200+门店可选。
大件商品（如显示器、椅子）使用德邦物流，免运费但配送时间较长。

## 常见问题
Q: 如何申请退款？
A: 在"我的订单"中找到对应订单，点击"申请退款"，填写原因后提交，客服将在24小时内审核。

Q: 发票如何开具？
A: 支持电子发票和纸质发票。电子发票在订单完成后自动发送到注册邮箱。纸质发票需联系客服申请，将在5个工作日内寄出。

Q: 优惠券如何使用？
A: 在结算页面选择可用优惠券即可抵扣。优惠券不可叠加使用，不可兑换现金。过期自动失效。
"""
        with open(docs_dir / "product_manual.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("RAG引擎：已创建默认产品手册")

    def load_directory(self, dir_path: str):
        """加载目录下所有文档"""
        loader = DirectoryLoader(dir_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        if not docs:
            print("RAG引擎：没有找到文档")
            return

        chunks = self.splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"RAG引擎：已加载{len(docs)}个文档，切分为{len(chunks)}个文本块")

    def search(self, query: str, k: int = 3) -> str:
        """检索相关文档，返回拼接的文本"""
        if not self.vectorstore:
            return "知识库为空，请先上传文档。"

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return "没有找到相关信息。"

        return "\n\n".join([doc.page_content for doc in results])