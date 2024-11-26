from datetime import datetime
from bson import ObjectId

class Document:
    def __init__(self, content, embedding, metadata):
        self.content = content
        self.embedding = embedding
        self.metadata = metadata
        self.metadata['uploadDate'] = datetime.utcnow()

    def to_dict(self):
        return {
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(data):
        return Document(
            content=data['content'],
            embedding=data['embedding'],
            metadata=data['metadata']
        )
