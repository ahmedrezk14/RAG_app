from openai import project

from .BaseController import BaseController
from models.db_schemes import Project,DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser):
        super().__init__()
        self.vectordb_client=vectordb_client
        self.generation_client=generation_client
        self.embedding_client=embedding_client
        self.template_parser=template_parser

    def create_collection_name(self, project_id:str):
        return f"collection_{project_id}".strip()

    def reset_vector_db_collection(self, project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project:Project):
        collection_name=self.create_collection_name(project_id=project.project_id)
        collection_info=self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(
            json.dumps(collection_info,default=lambda x:x.__dict__)
        )

    def index_into_vector_db(self, project:Project, chunks:List[DataChunk], chunk_ids:list[int], do_reset:bool=False):
        # 1 -> get collection name 
        collection_name=self.create_collection_name(project_id=project.project_id)

        # 2 -> hendel data 
        texts=[chunk.chunk_text for chunk in chunks]
        metadata=[chunk.chunk_metadata for chunk in chunks]
        vectors=[
            self.embedding_client.embed_text(
                text=text,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )
            for text in texts
        ]

        # 3 -> create collection 
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )
        
        # 4 -> insert in df 
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            record_ids=chunk_ids
        )

        return True

    def search_vector_db_collections(self, project:Project, text:str, limit:int=10):
        
        # 1-> get collection name 
        collection_name=self.create_collection_name(project_id=project.project_id)

        # 2-> text to embedding 
        vector=self.embedding_client.embed_text(text=text,document_type=DocumentTypeEnum.DOCUMENT.value)
        if not vector or len(vector)==0:
            return False

        # 3-> dp semantic search 
        results = self.vectordb_client.search_by_vector(collection_name=collection_name,vector=vector,limit=limit)

        if not results:
            return False
        
        return results

    def answer_rag_question(self, project:Project, query:str, limit:int=10):
        
        answer, full_prompt, chat_history=None,None,None

        # 1-> retrive relevant docs  
        retrived_documents=self.search_vector_db_collections(
            project=project,
            text=query,
            limit=limit
        )

        if not retrived_documents or len (retrived_documents) == 0:
            return answer, full_prompt, chat_history

        # 2-> build prompt

        sysyem_prompt=self.template_parser.get(group="rag", key="system_prompt", vars={})

        document_prompts="\n".join([
            self.template_parser.get(group="rag", key="document_prompt", vars={
                "doc_num":idx+1,
                "chunk_text":document.text
            })
            for idx,document in enumerate(retrived_documents)
        ])

        footer_prompt=self.template_parser.get(group="rag", key="footer_prompt", vars={})
        
        chat_history=[
            self.generation_client.construct_prompt(
                prompt=sysyem_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt="\n\n".join([document_prompts,footer_prompt])

        answer = self.generation_client.generate_text(prompt=full_prompt,chat_history=chat_history)

        return answer, full_prompt, chat_history



