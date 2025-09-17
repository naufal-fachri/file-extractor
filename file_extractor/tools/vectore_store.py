# Qdrant VectorStore
import os
import time
import uuid
import asyncio
import tiktoken
from loguru import logger
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Union, Any, Dict, Tuple
from contextlib import contextmanager

MAX_TOKENS_LENGTH = 8191 # maximum token length
DEFAULT_BATCH_SIZE = 100 # batch size for number of document example
MAX_TOKENS_PER_BATCH = DEFAULT_BATCH_SIZE * MAX_TOKENS_LENGTH
MAX_RETRIES = 3
RETRY_DELAY = 1.0
UPSERT_TIMEOUT = 20

def create_qdrant_collections(client: QdrantClient, collection_names: List[str], collection_configs: List[Dict[str, Any]]) -> List[str]:
    """Creating Qdrant collection based on provided collection name and collection configs"""

    zip_collection_names_configs = zip(collection_names, collection_configs)
    successful_collections_creation = []

    for name, config in zip_collection_names_configs:

        # check if collection exists.
        if client.collection_exists(collection_name=name):
            logger.warning(f"Collection {name} already exists. Skip creation.")
            continue

        try:
            # create collection.
            logger.info("Creating collection...")
            client.create_collection(
                collection_name=name,
                **config
            )
            logger.info(f"Collection {name} is sucessfully created.")
            successful_collections_creation.append(name)
        
        except Exception as e:
            raise e
        
    return successful_collections_creation


def delete_qdrant_collections(client: QdrantClient, collection_names: List[str]) -> List[str]:
    """Deleting Qdrant collection based on provided collection name"""

    successfull_colletions_deletion = []

    for name in collection_names:


        # collection exists.
        if not client.collection_exists(collection_name=name):
            logger.warning(f"Collection {name} doesn't exists in database.")
            continue

        try:
            client.delete_collection(
                collection_name=name,
                timeout=10,
            )
            logger.info(f"Collection {name} has been successfully deleted.")
            successfull_colletions_deletion.append(name)
        
        except Exception as e:
            raise e
        
    return successfull_colletions_deletion

class QdrantVectorStoreError(Exception):
    """Custom exception for QdrantVectorStore operations"""
    pass

class QdrantVectorStore:
    """Optimized QdrantVectorStore with improved logging and error handling"""
    
    DENSE_VECTOR_DEFAULT_NAME: str = "default"
    SPARSE_VECTOR_DEFAULT_NAME: str = "sparse"
    RETRIEVAL_MODE_DEFAULT: str = "dense"
    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        embeddings: Embeddings,
        vector_name: str = DENSE_VECTOR_DEFAULT_NAME,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        retrieval_mode: str = RETRIEVAL_MODE_DEFAULT,
        distance: models.Distance = models.Distance.COSINE,
        sparse_embedding: Optional[Embeddings] = None,
        sparse_vector_name: Optional[str] = None,
    ):
        """
        Initialize QdrantVectorStore
        
        Args:
            max_workers: Number of worker threads for parallel processing
        """
        # Validate inputs
        if retrieval_mode not in ["dense", "sparse", "hybrid"]:
            raise ValueError(f"Invalid retrieval_mode: {retrieval_mode}")
        
        if retrieval_mode in ["sparse", "hybrid"] and sparse_embedding is None:
            raise ValueError(f"sparse_embedding is required for {retrieval_mode} mode")
        
        if retrieval_mode in ["sparse", "hybrid"] and sparse_vector_name is None:
            raise ValueError(f"sparse_vector_name is required for {retrieval_mode} mode")

        # Configuration
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.retrieval_mode = retrieval_mode
        self.distance = distance
        self.vector_name = vector_name
        self.sparse_embedding = sparse_embedding
        self.sparse_vector_name = sparse_vector_name
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key

        logger.info(f"Initialized QdrantVectorStore - Collection: {collection_name}")

    def _tokenize(self, text: str, model_name: str="gpt-4") -> int:
        """Count tokens in text using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text=text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Failed to tokenize text, using character count: {e}")
            return len(text) // 4  # Rough approximation

    def _prepare_batches(self, documents: List[Document], batch_size: int) -> List[List[Document]]:
        """
        Optimize document batching with better memory efficiency and logging
        """
        if not documents:
            logger.warning("Empty document list provided for batching")
            return []

        logger.debug(f"Preparing batches for {len(documents)} documents with batch_size={batch_size}")
        
        batches: List[List[Document]] = []
        current_batch: List[Document] = []
        current_batch_tokens = 0
        
        for doc_idx, document in enumerate(documents):
            try:
                doc_tokens = self._tokenize(document.page_content)
                
                # Handle oversized documents
                if doc_tokens > MAX_TOKENS_PER_BATCH:
                    logger.warning(
                        f"Document {doc_idx} has {doc_tokens} tokens, "
                        f"exceeding MAX_TOKENS_PER_BATCH ({MAX_TOKENS_PER_BATCH}). "
                        "Processing separately."
                    )
                    # Flush current batch if not empty
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_batch_tokens = 0
                    
                    # Add oversized document as single batch
                    batches.append([document])
                    continue
                
                # Check if adding this document would exceed limits
                would_exceed_tokens = current_batch_tokens + doc_tokens > MAX_TOKENS_PER_BATCH
                would_exceed_size = len(current_batch) >= batch_size
                
                if (would_exceed_tokens or would_exceed_size) and current_batch:
                    batches.append(current_batch)
                    current_batch = [document]
                    current_batch_tokens = doc_tokens
                else:
                    current_batch.append(document)
                    current_batch_tokens += doc_tokens
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {e}")
                raise QdrantVectorStoreError(f"Failed to process document {doc_idx}: {e}")
        
        # Add remaining documents
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches from {len(documents)} documents")
        return batches

    @contextmanager
    def _timer(self, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield

        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{operation} completed in {elapsed:.2f}s")

    def generate_point_id(self, file_name: str, user_id: str, chat_id: str, doc_idx: str) -> str:
        """Generate deterministic point ID"""
        content = f"{user_id}_{chat_id}_{file_name}_{doc_idx}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

    async def async_generate_points_dense(self, batch: List[Document]) -> List[models.PointStruct]:
        """Generate dense vector points with improved error handling"""

        with self._timer(f"Dense embedding generation for batch of {len(batch)}"):
            try:
                contents = [doc.page_content for doc in batch]
                vectors = await self.embeddings.aembed_documents(texts=contents, chunk_size=len(batch))
                
                if len(batch) != len(vectors):
                    raise QdrantVectorStoreError(
                        f"Batch size mismatch: {len(batch)} documents vs {len(vectors)} vectors"
                    )
                
                points = []
                for idx, (doc, vector) in enumerate(zip(batch, vectors)):
                    point = models.PointStruct(
                        id=self.generate_point_id(file_name=doc.metadata["file_name"],
                                                  user_id=doc.metadata["user_id"],
                                                  chat_id=doc.metadata["chat_id"],
                                                  doc_idx=idx),
                        payload={
                            self.content_payload_key: doc.page_content,
                            self.metadata_payload_key: doc.metadata
                        },
                        vector={self.vector_name: vector}
                    )
                    points.append(point)
                
                logger.debug(f"Generated {len(points)} dense points")
                return points
                
            except Exception as e:
                logger.error(f"Failed to generate dense points: {e}")
                raise QdrantVectorStoreError(f"Dense point generation failed: {e}")

    async def async_generate_points_sparse(self, batch: List[Document]) -> List[models.PointStruct]:
        """Generate sparse vector points with improved error handling"""
        with self._timer(f"Sparse embedding generation for batch of {len(batch)}"):
            try:
                contents = [doc.page_content for doc in batch]
                vectors = await self.sparse_embedding.aembed_documents(texts=contents, chunk_size=len(batch))
                
                if len(batch) != len(vectors):
                    raise QdrantVectorStoreError(
                        f"Batch size mismatch: {len(batch)} documents vs {len(vectors)} vectors"
                    )
                
                points = []
                for idx, (doc, vector) in enumerate(zip(batch, vectors)):
                    point = models.PointStruct(
                        id=self.generate_point_id(file_name=doc.metadata["file_name"],
                                                  user_id=doc.metadata["user_id"],
                                                  chat_id=doc.metadata["chat_id"],
                                                  doc_idx=idx),
                        payload={
                            self.content_payload_key: doc.page_content,
                            self.metadata_payload_key: doc.metadata
                        },
                        vector={
                            self.sparse_vector_name: models.SparseVector(
                                indices=vector.indices, 
                                values=vector.values
                            )
                        }
                    )
                    points.append(point)
                
                logger.debug(f"Generated {len(points)} sparse points")
                return points
                
            except Exception as e:
                logger.error(f"Failed to generate sparse points: {e}")
                raise QdrantVectorStoreError(f"Sparse point generation failed: {e}")

    async def async_generate_points_hybrid(self, batch: List[Document]) -> List[models.PointStruct]:
        """Generate hybrid vector points with improved error handling"""
        with self._timer(f"Hybrid embedding generation for batch of {len(batch)}"):
            try:
                contents = [doc.page_content for doc in batch]
                
                # Generate embeddings concurrently if possible
                dense_vectors = await self.embeddings.aembed_documents(texts=contents, chunk_size=len(batch))
                sparse_vectors = await self.sparse_embedding.aembed_documents(texts=contents, chunk_size=len(batch))
                
                if len(batch) != len(dense_vectors) or len(batch) != len(sparse_vectors):
                    raise QdrantVectorStoreError(
                        f"Batch size mismatch: {len(batch)} documents vs "
                        f"{len(dense_vectors)} dense vectors vs {len(sparse_vectors)} sparse vectors"
                    )
                
                points = []
                for idx, (doc, dense_vec, sparse_vec) in enumerate(zip(batch, dense_vectors, sparse_vectors)):
                    point = models.PointStruct(
                        id=self.generate_point_id(file_name=doc.metadata["file_name"],
                                                  user_id=doc.metadata["user_id"],
                                                  chat_id=doc.metadata["chat_id"],
                                                  doc_idx=idx),
                        payload={
                            self.content_payload_key: doc.page_content,
                            self.metadata_payload_key: doc.metadata
                        },
                        vector={
                            self.vector_name: dense_vec,
                            self.sparse_vector_name: models.SparseVector(
                                indices=sparse_vec.indices,
                                values=sparse_vec.values
                            )
                        }
                    )
                    points.append(point)
                
                logger.debug(f"Generated {len(points)} hybrid points")
                return points
                
            except Exception as e:
                logger.error(f"Failed to generate hybrid points: {e}")
                raise QdrantVectorStoreError(f"Hybrid point generation failed: {e}")

    async def async_upsert_with_retry(self, points: List[models.PointStruct], batch_idx: int) -> bool:
        """Upsert points with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                with self._timer(f"Upsert batch {batch_idx} (attempt {attempt + 1})"):
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True  # Changed to True for production reliability
                    )
                logger.debug(f"Successfully upserted batch {batch_idx} ({len(points)} points)")
                return True
                
            except Exception as e:
                logger.warning(
                    f"Upsert attempt {attempt + 1}/{MAX_RETRIES} failed for batch {batch_idx}: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to upsert batch {batch_idx} after {MAX_RETRIES} attempts")
                    raise QdrantVectorStoreError(f"Upsert failed for batch {batch_idx}: {e}")
        
        return False


    async def aadd_documents(self, documents: List[Document], batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Async version of add_documents with improved concurrency
        """
        if not documents:
            logger.warning("No documents provided to add")
            return

        start_time = time.time()
        logger.info(f"Starting async add of {len(documents)} documents to collection '{self.collection_name}'")
        
        try:
            batches = self._prepare_batches(documents, batch_size)
            if not batches:
                logger.warning("No batches created from documents")
                return

            # Process all batches concurrently
            async def process_batch(batch_idx: int, batch: List[Document]) -> Tuple[int, int]:
                try:
                    logger.debug(f"Processing async batch {batch_idx + 1}/{len(batches)} ({len(batch)} documents)")
                    
                    if self.retrieval_mode == "dense":
                        points = await self.as_generate_points_dense(batch)

                    elif self.retrieval_mode == "sparse":
                        points = await self.a_generate_points_sparse(batch)

                    elif self.retrieval_mode == "hybrid":
                        points = await self.a_generate_points_hybrid(batch)

                    else:
                        raise QdrantVectorStoreError(f"Invalid retrieval mode: {self.retrieval_mode}")
                    
                    await self.async_upsert_with_retry(points, batch_idx)
                    return batch_idx, len(points)
                    
                except Exception as e:
                    logger.error(f"Failed to process async batch {batch_idx}: {e}")
                    raise

            # Execute all batches concurrently
            tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_batches = 0
            total_points = 0
            errors = []
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(result)
                else:
                    successful_batches += 1
                    total_points += result[1]

            elapsed_time = time.time() - start_time
            
            if errors:
                logger.error(f"Failed to process {len(errors)} batches: {errors}")
                raise QdrantVectorStoreError(f"{len(errors)} batches failed during async processing")
            else:
                logger.info(
                    f"Successfully added all {total_points} points from {len(batches)} batches "
                    f"to collection '{self.collection_name}' in {elapsed_time:.2f}s (async)"
                )

        except Exception as e:
            logger.error(f"Critical error in async add_documents: {e}")
            raise QdrantVectorStoreError(f"Failed to async add documents: {e}")

    def _document_from_point(
        self,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        """Convert Qdrant point to Document with enhanced metadata"""
        metadata = scored_point.payload.get(metadata_payload_key, {}).copy()
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        metadata["_score"] = getattr(scored_point, 'score', None)

        return Document(
            page_content=scored_point.payload.get(content_payload_key, ""),
            metadata=metadata,
        )

    # Async search methods with similar improvements
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        hybrid_fusion: Optional[models.FusionQuery] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async similarity search"""
        results = await self.asimilarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            hybrid_fusion=hybrid_fusion,
            **kwargs,
        )
        return [doc for doc, _ in results]
    
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[models.ReadConsistency] = None,
        hybrid_fusion: Optional[models.FusionQuery] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async similarity search with scores"""
        try:
            query_options = {
                "collection_name": self.collection_name,
                "query_filter": filter,
                "search_params": search_params,
                "limit": k,
                "offset": offset,
                "with_payload": True,
                "with_vectors": False,
                "score_threshold": score_threshold,
                "consistency": consistency,
                **kwargs,
            }

            if self.retrieval_mode == "dense":
                query_embedding = await self.embeddings.aembed_query(query)
                results = await self.client.query_points(
                    query=query_embedding,
                    using=self.vector_name,
                    **query_options,
                ).points

            elif self.retrieval_mode == "sparse":
                query_embedding = await self.sparse_embedding.aembed_query(query)
                results = await self.client.query_points(
                    query=models.SparseVector(
                        indices=query_embedding.indices,
                        values=query_embedding.values,
                    ),
                    using=self.sparse_vector_name,
                    **query_options,
                ).points

            elif self.retrieval_mode == "hybrid":
                query_dense = await self.embeddings.aembed_query(query)
                query_sparse = await self.sparse_embedding.aembed_query(query)
                
                results = await self.client.query_points(
                    prefetch=[
                        models.Prefetch(
                            using=self.vector_name,
                            query=query_dense,
                            filter=filter,
                            limit=k,
                            params=search_params,
                        ),
                        models.Prefetch(
                            using=self.sparse_vector_name,
                            query=models.SparseVector(
                                indices=query_sparse.indices,
                                values=query_sparse.values,
                            ),
                            filter=filter,
                            limit=k,
                            params=search_params,
                        ),
                    ],
                    query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
                    **query_options,
                ).points

            else:
                raise QdrantVectorStoreError(f"Invalid retrieval mode: {self.retrieval_mode}")

            # Convert results to documents with scores
            doc_score_pairs = [
                (
                    self._document_from_point(
                        result,
                        self.collection_name,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                    result.score,
                )
                for result in results
            ]
            
            logger.debug(f"Found {len(doc_score_pairs)} results for query")
            return doc_score_pairs

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise QdrantVectorStoreError(f"Similarity search failed: {e}")