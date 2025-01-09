from langchain.docstore.document import Document


class dbs:

    def __init__(self, chrdb=[]):

        self.ids = []
        self.embeds = []
        self.metadatas = []
        self.docs = []
        self.vis = set()
        if isinstance(chrdb, list):
            pass

        else:
            data = chrdb.get(include=["embeddings", "metadatas", "documents"])
            if "ids" in data:
                self.ids = data["ids"]
                self.vis = set(data["ids"])
            if "embeddings" in data:
                self.embeds = data["embeddings"]
            else:
                self.embeds = [[] for _ in range(len(data["ids"]))]
            if "metadatas" in data:
                self.metadatas = data["metadatas"]
            else:
                self.metadatas = [{} for _ in range(len(data["ids"]))]
            if "documents" in data:
                self.docs = data["documents"]
            else:
                self.docs = ["" for _ in range(len(data["ids"]))]

    def merge(self, db: 'dbs'):

        for i in range(len(db.ids)):
            if db.ids[i] not in self.vis:
                self.ids.append(db.ids[i])
                self.embeds.append(db.embeds[i])
                self.metadatas.append(db.metadatas[i])
                self.docs.append(db.docs[i])
                self.vis.add(db.ids[i])
        # self.ids.extend(db.ids)
        # self.embeds.extend(db.embeds)
        # self.metadatas.extend(db.metadatas)
        # self.docs.extend(db.docs)

    def add_chromadb(self, chrdb):
        data = chrdb.get(include=["embeddings", "metadatas", "documents"])
        if "ids" in data:
            self.ids.extend(data["ids"])

        if "embeddings" in data:
            self.embeds.extend(data["embeddings"])
        else:
            self.embeds.extend([[] for _ in range(len(data["ids"]))])
        if "metadatas" in data:
            self.metadatas.extend(data["metadatas"])
        else:
            self.metadatas.extend([{} for _ in range(len(data["ids"]))])
        if "documents" in data:
            self.docs.extend(data["documents"])
        else:
            self.docs.extend(["" for _ in range(len(data["ids"]))])

    def get_Documents(self):
        return [
            Document(page_content=self.docs[i], metadata=self.metadatas[i])
            for i in range(len(self.docs))
        ]

    def get_docs(self):
        return self.docs

    def get_ids(self):
        return self.ids

    def get_metadatas(self):
        return self.metadatas

    def get_embeds(self):
        return self.embeds
