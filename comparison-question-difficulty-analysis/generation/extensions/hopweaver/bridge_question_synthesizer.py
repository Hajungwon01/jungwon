import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.config import Config
from hopweaver.components.utils.data_reader import DocumentReader
from hopweaver.components.bridge.bridge_entity_extractor import EntityExtractor
from hopweaver.components.bridge.bridge_retriever import RetrieverWrapper, DiverseRetrieverWrapper, RerankRetrieverWrapper
from hopweaver.components.bridge.bridge_question_generator import QuestionGenerator
from hopweaver.components.bridge.bridge_polisher import Polisher

class QuestionSynthesizer:
    """
    Multi-hop question synthesizer main class that integrates document reading,
    entity extraction, document retrieval, and question generation.
    """
    def __init__(self, config_path, lambda1=None, lambda2=None, lambda3=None):
        """
        Initialize the question synthesizer
        
        Args:
            config_path (str): Path to configuration file
            lambda1 (float, optional): Query relevance weight
            lambda2 (float, optional): Source document diversity weight
            lambda3 (float, optional): Selected document set diversity weight
        """
        # Load configuration
        self.config = Config(config_path, {})
        
        # Set GPU environment variable
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"])
        
        # Initialize components
        self.doc_reader = DocumentReader(self.config['corpus_path'])
        self.entity_extractor = EntityExtractor(self.config)
        
        # Select retriever based on configuration
        self.retriever_type = self.config["retriever_type"] if "retriever_type" in self.config else "diverse"  # Default: diverse retriever
        
        # Use provided lambda parameters first, then config values, or defaults
        l1 = lambda1 if lambda1 is not None else (self.config["lambda1"] if "lambda1" in self.config else 0.7)
        l2 = lambda2 if lambda2 is not None else (self.config["lambda2"] if "lambda2" in self.config else 0.1)
        l3 = lambda3 if lambda3 is not None else (self.config["lambda3"] if "lambda3" in self.config else 0.2)
        
        if self.retriever_type == "rerank":
            # Use rerank retriever
            reranker_path = self.config["reranker_path"] if "reranker_path" in self.config else "./models/bge-reranker-v2-m3"
            self.retriever = RerankRetrieverWrapper(
                self.config, 
                reranker_path=reranker_path,
                lambda1=l1, 
                lambda2=l2, 
                lambda3=l3,
                use_fp16=self.config["use_fp16"] if "use_fp16" in self.config else True
            )
            print(f"Using rerank retriever: RerankRetrieverWrapper, reranker model path: {reranker_path}")
        elif self.retriever_type == "standard":
            # Use standard retriever
            self.retriever = RetrieverWrapper(self.config)
            print("Using standard retriever: RetrieverWrapper")
        else:
            # Use diversity retriever
            self.retriever = DiverseRetrieverWrapper(
                self.config, 
                lambda1=l1, 
                lambda2=l2, 
                lambda3=l3
            )
            print("Using diversity retriever: DiverseRetrieverWrapper")
        
        self.question_generator = QuestionGenerator(self.config)
        self.polisher = Polisher(self.config)
        
        # Configure output directory
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_3hop_question(
        self,
        save_output: bool = True,
        verbose: bool = True,
        max_entities_per_doc: int = 3,
        max_docs_per_entity: int = 5,
    ):
        """
        3-hop 전용 질문 생성기.

        흐름:
        1) 문서 A 선택
        2) 문서 A에서 bridge entity 1 추출
        3) bridge entity 1로 문서 B 검색
        4) 문서 B에서 bridge entity 2 추출
        5) bridge entity 2로 문서 C 검색
        6) (A, B, C, entity1, entity2)를 사용해
           QuestionGenerator.generate_3_sub_questions + synthesize_3_hop_question 호출
        """

        try:
            print("=" * 60)
            print("Starting 3-Hop Question Generation (A -> B -> C)")
            print("=" * 60)

            # ------------------------------
            # Step 1. 문서 A 선택
            # ------------------------------
            print("\nStep 1: Selecting source document (Document A)")
            source_docs = self.doc_reader.get_heuristic_documents(count=1, min_length=300)
            if not source_docs:
                print("Failed to retrieve source document")
                return None

            doc_a = source_docs[0]
            if verbose:
                print(f"✓ Document A selected: ID={doc_a.get('id','')}, "
                      f"Title={doc_a.get('title','No Title')}")
                prev = doc_a.get("contents", "")[:300]
                print("Preview:")
                print(prev + ("..." if len(doc_a.get("contents", "")) > 300 else ""))

            # ------------------------------
            # Step 2. 문서 A에서 bridge entity 1 추출
            # ------------------------------
            print("\nStep 2: Extracting bridge entities from Document A")
            entities_a = self.entity_extractor.extract_entities(doc_a)
            if not entities_a:
                print("✗ No entities extracted from Document A")
                return None

            entities_a = entities_a[:max_entities_per_doc]

            # 문서 체인 / 엔티티 체인 초기화
            documents_chain = [doc_a]   # [A, B, C]
            bridge_entities = []        # [entity1, entity2]

            # A에서 추출된 엔티티들을 순차적으로 시도
            for e1_idx, entity1 in enumerate(entities_a, start=1):
                name1 = entity1.get("name", "")
                type1 = entity1.get("type", "")
                query1 = entity1.get("query", name1)
                segs1_list = entity1.get("segments", []) or []
                segs1 = " ".join(segs1_list)
                doc_a_id = doc_a.get("id", "")

                if not (name1 and type1 and query1 and segs1 and doc_a_id):
                    print(f"  [Entity1 {e1_idx}] Incomplete entity info, skip")
                    continue

                if verbose:
                    print(f"\n[Entity1 {e1_idx}] name={name1}, type={type1}")
                    if segs1_list:
                        seg_prev = segs1_list[0]
                        if len(seg_prev) > 200:
                            seg_prev = seg_prev[:200] + "..."
                        print("  segment preview:", seg_prev)

                # ------------------------------
                # Step 3. entity1로 문서 B 검색
                # ------------------------------
                print("  Step 3: Retrieving candidate documents for bridge entity 1 (Document B)")
                if isinstance(self.retriever, RerankRetrieverWrapper) and hasattr(self.retriever, "retrieve_with_rerank"):
                    retrieved_b = self.retriever.retrieve_with_rerank(
                        query1,
                        segs1,
                        top_k=max_docs_per_entity,
                        doc_id=doc_a_id,
                    )
                else:
                    retrieved_b = self.retriever.retrieve_with_diversity(
                        query1,
                        segs1,
                        top_k=max_docs_per_entity,
                        doc_id=doc_a_id,
                    )

                if not retrieved_b:
                    print("    ✗ No documents retrieved for entity 1")
                    continue

                used_ids = {doc_a_id}
                candidate_bs = [d for d in retrieved_b if d.get("id") not in used_ids]
                if not candidate_bs:
                    print("    ✗ All retrieved docs already used (A)")
                    continue

                # B 후보 문서를 하나씩 시도
                for b_idx, doc_b in enumerate(candidate_bs[:max_docs_per_entity], start=1):
                    if verbose:
                        print(f"\n    [Candidate B {b_idx}] ID={doc_b.get('id','')}, "
                              f"Title={doc_b.get('title','No Title')}")
                        prev_b = doc_b.get("contents", "")[:200]
                        print("      preview:", prev_b + ("..." if len(doc_b.get("contents", "")) > 200 else ""))

                    # ------------------------------
                    # Step 4. 문서 B에서 bridge entity 2 추출
                    # ------------------------------
                    print("    Step 4: Extracting bridge entities from Document B")
                    entities_b = self.entity_extractor.extract_entities(doc_b)
                    if not entities_b:
                        print("      ✗ No entities extracted from Document B")
                        continue

                    entities_b = entities_b[:max_entities_per_doc]

                    for e2_idx, entity2 in enumerate(entities_b, start=1):
                        name2 = entity2.get("name", "")
                        type2 = entity2.get("type", "")
                        query2 = entity2.get("query", name2)
                        segs2_list = entity2.get("segments", []) or []
                        segs2 = " ".join(segs2_list)
                        doc_b_id = doc_b.get("id", "")

                        # entity1과 entity2가 완전히 같은 문자열이면 스킵 (너무 자명한 경우 회피)
                        if name2 == name1:
                            print(f"      [Entity2 {e2_idx}] Same as entity1, skip")
                            continue

                        if not (name2 and type2 and query2 and segs2 and doc_b_id):
                            print(f"      [Entity2 {e2_idx}] Incomplete entity info, skip")
                            continue

                        if verbose:
                            print(f"\n      [Entity2 {e2_idx}] name={name2}, type={type2}")
                            if segs2_list:
                                seg2_prev = segs2_list[0]
                                if len(seg2_prev) > 200:
                                    seg2_prev = seg2_prev[:200] + "..."
                                print("        segment preview:", seg2_prev)

                        # ------------------------------
                        # Step 5. entity2로 문서 C 검색
                        # ------------------------------
                        print("      Step 5: Retrieving candidate documents for bridge entity 2 (Document C)")
                        if isinstance(self.retriever, RerankRetrieverWrapper) and hasattr(self.retriever, "retrieve_with_rerank"):
                            retrieved_c = self.retriever.retrieve_with_rerank(
                                query2,
                                segs2,
                                top_k=max_docs_per_entity,
                                doc_id=doc_b_id,
                            )
                        else:
                            retrieved_c = self.retriever.retrieve_with_diversity(
                                query2,
                                segs2,
                                top_k=max_docs_per_entity,
                                doc_id=doc_b_id,
                            )

                        if not retrieved_c:
                            print("        ✗ No documents retrieved for entity 2")
                            continue

                        used_ids_c = {doc_a_id, doc_b_id}
                        candidate_cs = [d for d in retrieved_c if d.get("id") not in used_ids_c]
                        if not candidate_cs:
                            print("        ✗ All retrieved docs already used (A/B)")
                            continue

                        # C 후보 문서를 하나씩 시도
                        for c_idx, doc_c in enumerate(candidate_cs[:max_docs_per_entity], start=1):
                            if verbose:
                                print(f"\n        [Candidate C {c_idx}] ID={doc_c.get('id','')}, "
                                      f"Title={doc_c.get('title','No Title')}")
                                prev_c = doc_c.get("contents", "")[:200]
                                print("          preview:", prev_c + ("..." if len(doc_c.get("contents", "")) > 200 else ""))

                            # ------------------------------
                            # Step 6. 3-hop 하위질문 생성
                            # ------------------------------
                            if not hasattr(self.question_generator, "generate_3_sub_questions"):
                                print("QuestionGenerator.generate_3_sub_questions is not implemented.")
                                return None

                            # Document A 의 relevant segment: entity1.segments 가 있으면 그걸 사용
                            doc_a_segments = " ".join(segs1_list) or doc_a.get("contents", "")
                            doc_b_document = doc_b.get("contents", "")
                            doc_c_document = doc_c.get("contents", "")

                            sub_result = self.question_generator.generate_3_sub_questions(
                                bridge_entity1=name1,
                                bridge_entity2=name2,
                                entity_type1=type1,
                                entity_type2=type2,
                                doc_a_segments=doc_a_segments,
                                doc_b_document=doc_b_document,
                                doc_c_document=doc_c_document,
                            )

                            if not sub_result or not sub_result.get("sub_questions"):
                                print("          ✗ Failed to generate 3-hop sub-questions, try next C")
                                continue

                            # ------------------------------
                            # Step 7. 3-hop multi-hop 질문 합성
                            # ------------------------------
                            if not hasattr(self.question_generator, "synthesize_3_hop_question"):
                                print("QuestionGenerator.synthesize_3_hop_question is not implemented.")
                                return None

                            documents_chain = [doc_a, doc_b, doc_c]
                            bridge_entities = [entity1, entity2]
                            hop_sub_results = [sub_result]

                            multi_hop_result = self.question_generator.synthesize_3_hop_question(
                                hop_sub_results=hop_sub_results,
                                documents=documents_chain,
                                bridge_entities=bridge_entities,
                            )

                            if not multi_hop_result:
                                print("          ✗ Failed to synthesize 3-hop question, try next C")
                                continue

                            # ------------------------------
                            # Step 8. Polisher로 다듬기
                            # ------------------------------
                            mh_q = multi_hop_result.get("multi_hop_question", "")
                            mh_a = multi_hop_result.get("answer", "")
                            mh_reason = multi_hop_result.get("reasoning_path", "")

                            polish_result = None
                            if hasattr(self.polisher, "polish_n_hop_question"):
                                polish_result = self.polisher.polish_n_hop_question(
                                    multi_hop_question=mh_q,
                                    answer=mh_a,
                                    reasoning_path=mh_reason,
                                    hop_sub_results=hop_sub_results,
                                    documents=documents_chain,
                                )

                            # ------------------------------
                            # 최종 결과 dict 구성 및 저장
                            # ------------------------------
                            # ------------------------------
                            # 최종 결과 dict 구성 및 저장
                            # ------------------------------
                            result = {
                                "hop_count": 3,
                                # 3개 문서 체인 (문서 리스트로도 저장)
                                "documents": [
                                    {
                                        "order": 0,
                                        "id": doc_a.get("id", ""),
                                        "title": doc_a.get("title", ""),
                                        "contents": doc_a.get("contents", ""),
                                    },
                                    {
                                        "order": 1,
                                        "id": doc_b.get("id", ""),
                                        "title": doc_b.get("title", ""),
                                        "contents": doc_b.get("contents", ""),
                                    },
                                    {
                                        "order": 2,
                                        "id": doc_c.get("id", ""),
                                        "title": doc_c.get("title", ""),
                                        "contents": doc_c.get("contents", ""),
                                    },
                                ],
                                # 개별 필드는 기존대로 유지 (혹시 다른 코드에서 쓸 수 있으니)
                                "source_doc": {
                                    "id": doc_a.get("id", ""),
                                    "title": doc_a.get("title", ""),
                                    "content": doc_a.get("contents", ""),
                                },
                                "mid_doc": {
                                    "id": doc_b.get("id", ""),
                                    "title": doc_b.get("title", ""),
                                    "content": doc_b.get("contents", ""),
                                },
                                "target_doc": {
                                    "id": doc_c.get("id", ""),
                                    "title": doc_c.get("title", ""),
                                    "content": doc_c.get("contents", ""),
                                },
                                "bridge_entities": [
                                    {
                                        "order": 1,
                                        "name": name1,
                                        "type": type1,
                                        "segments": segs1_list,
                                        "query": query1,
                                    },
                                    {
                                        "order": 2,
                                        "name": name2,
                                        "type": type2,
                                        "segments": segs2_list,
                                        "query": query2,
                                    },
                                ],
                                "hop_sub_results": hop_sub_results,
                                "multi_hop_question": multi_hop_result,
                                "polish_result": polish_result,
                            }

                            if save_output:
                                # 기존 _save_result 재사용 (hop_count만 3으로 들어감)
                                self._save_result(result, mode="single")

                            print("\n✓ 3-hop question successfully generated!")
                            return result

                        # C 후보 다 실패 → 다음 entity2
                        print("      ↺ No valid 3-hop chain found for this entity 2, try next")

                    # B 후보에서 유효한 C를 못 찾음 → 다음 B
                    print("    ↺ No valid 3-hop chain found for this Document B, try next")

                # entity1로 유효 chain 없음 → 다음 entity1
                print(f"↺ No valid 3-hop chain found for Entity1 {e1_idx}, try next")

            print("✗ Failed to build a valid 3-hop bridge chain")
            return None

        except Exception as e:
            import traceback
            print("Exception during 3-hop question generation:")
            print(e)
            traceback.print_exc()
            return None
        
    def _save_result(self, result, mode="single", jsonl_file=None):
        if mode == "single":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            question_str = result.get("multi_hop_question", {}).get("multi_hop_question", "")
            question_id = "".join(e for e in question_str[:10] if e.isalnum())
            base = f"multihop_h{result.get('hop_count', 3)}_{timestamp}_{question_id}"

            json_path = os.path.join(self.output_dir, base + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            txt_path = os.path.join(self.output_dir, base + "_summary.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=== 3-Hop Multi-hop Question Generation Summary ===\n\n")
                f.write(f"Hop count: {result.get('hop_count', 3)}\n\n")

                # Documents
                f.write("=== Documents ===\n")
                for d in result.get("documents", []):
                    f.write(f"- Doc order {d['order']}: ID={d['id']}, Title={d['title']}\n")
                f.write("\n")

                # Bridge Entities (여러 개)
                f.write("=== Bridge Entities ===\n")
                bridge_entities = result.get("bridge_entities", [])
                if not bridge_entities:
                    f.write("No bridge entities found.\n\n")
                else:
                    for be in bridge_entities:
                        order = be.get("order", "?")
                        name = be.get("name", "")
                        etype = be.get("type", "")
                        f.write(f"[Hop {order}] {name} (type={etype})\n")
                    f.write("\n")

                # Sub-questions
                f.write("=== Sub-questions ===\n")
                for hop_idx, sub_res in enumerate(result.get("hop_sub_results", []), start=1):
                    f.write(f"\n--- Hop {hop_idx} ---\n")
                    analysis = sub_res.get("analysis", {})
                    if analysis:
                        f.write(f"Bridge connection 1: {analysis.get('bridge_connection1', '')}\n")
                        f.write(f"Bridge connection 2: {analysis.get('bridge_connection2', '')}\n")
                        f.write(f"Doc A seg: {analysis.get('doc_a_seg', '')}\n")
                        f.write(f"Doc B seg: {analysis.get('doc_b_seg', '')}\n")
                        f.write(f"Doc C seg: {analysis.get('doc_c_seg', '')}\n")
                        f.write(f"Reasoning path 1: {analysis.get('reasoning_path1', '')}\n")
                        f.write(f"Reasoning path 2: {analysis.get('reasoning_path2', '')}\n")

                    for i, q in enumerate(sub_res.get("sub_questions", []), start=1):
                        f.write(f"\nSub-question {i}: {q.get('question','')}\n")
                        f.write(f"Answer: {q.get('answer','')}\n")
                        f.write(f"Source: {q.get('source','')}\n")

                # Final multi-hop question
                f.write("\n=== Final Multi-hop Question ===\n")
                mh = result.get("multi_hop_question", {})
                f.write(f"Question: {mh.get('multi_hop_question','')}\n")
                f.write(f"Answer: {mh.get('answer','')}\n")
                f.write(f"Reasoning path: {mh.get('reasoning_path','')}\n")
                f.write(f"Sources: {mh.get('sources','')}\n")

                # Polish
                f.write("\n=== Polish Result ===\n")
                pr = result.get("polish_result")
                if pr:
                    f.write(f"Status: {pr.get('status','')}\n")
                    f.write(f"Refined question: {pr.get('refined_question','')}\n")
                    f.write(f"Refined answer: {pr.get('answer','')}\n")
                    f.write(f"Refined reasoning: {pr.get('refined_reasoning_path','')}\n")
                else:
                    f.write("No polish or polish failed.\n")

            print(f"Result saved to:\n  JSON: {json_path}\n  TXT : {txt_path}")

        elif mode == "batch":
            if not jsonl_file:
                raise ValueError("jsonl_file path must be provided in batch mode")
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unknown save mode: {mode}")
    
    def batch_generate(self, count=5, save_all=True):
        """Batch generate multi-hop questions
        
        Args:
            count (int): Number of questions to generate
            save_all (bool): Whether to save all generated results
            
        Returns:
            list: List of successfully generated questions
        """
        results = []
        success_count = 0
        attempt_count = 0
        max_attempts = count * 3  # Set maximum attempts to avoid infinite loop
        
        print(f"Starting batch generation of {count} multi-hop questions...")
        
        # Create jsonl output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_output_file = os.path.join(self.output_dir, f"multihop_batch_{timestamp}.jsonl")
        
        while success_count < count and attempt_count < max_attempts:
            attempt_count += 1
            print(f"\n===== Generation attempt {attempt_count}/{max_attempts} (Success: {success_count}/{count}) =====")
            
            # Generate question, but do not save (we will handle saving here)
            result = self.generate_3hop_question(save_output=False)
            
            if result:
                # If save_all, append result to jsonl file
                if save_all:
                    self._save_result(result, mode="batch", jsonl_file=jsonl_output_file)
                
                results.append(result)
                success_count += 1
                print(f"Successfully generated {success_count} questions, stopping generation")
            else:
                print("Generation failed, trying next...")
        
        print(f"\nBatch generation complete: Total attempts {attempt_count}, successful {success_count}")
        
        if save_all and success_count > 0:
            print(f"All results saved to: {jsonl_output_file}")
        
        return results


if __name__ == "__main__":
    # Configuration file path
    config_path = "/home/dilab/Desktop/Jiwon/naver_experiment/HopWeaver/hopweaver/config_lib/my_config.yaml"
    
    # Create question synthesizer
    synthesizer = QuestionSynthesizer(config_path)
    
    # Single question generation mode
    single_mode = False  # True: single, False: batch
    
    if single_mode:
        print("=== Single Question Generation Mode (3-hop) ===")
        # 3-hop 질문 생성
        result = synthesizer.generate_3hop_question(save_output=True)
        
        if result is None:
            print("\nGeneration failed")
        else:
            print("\nGeneration successful!\n")

            # ============================
            # 1) 문서 정보
            # ============================
            print("=== Documents in Chain ===")
            docs = result.get("documents", [])
            for d in docs:
                print(f"  [Doc {d.get('order','?')}] ID={d.get('id','')}, Title={d.get('title','')}")

            # ============================
            # 2) 브릿지 엔티티 정보
            # ============================
            print("\n=== Bridge Entities ===")
            bridge_entities = result.get("bridge_entities", [])
            if not bridge_entities:
                print("  (No bridge entities)")
            else:
                for be in bridge_entities:
                    order = be.get("order", "?")
                    name = be.get("name", "")
                    etype = be.get("type", "")
                    print(f"  [Hop {order}] {name} (type={etype})")
                    # segments 몇 개만 프린트
                    segs = be.get("segments", [])
                    for i, seg in enumerate(segs[:2]):
                        seg_short = seg if len(seg) <= 200 else seg[:200] + "..."
                        print(f"    Segment {i+1}: {seg_short}")
            
            # ============================
            # 3) 하위 질문 / 분석
            # ============================
            print("\n=== Sub-questions & Analysis ===")
            hop_sub_results = result.get("hop_sub_results", [])
            if not hop_sub_results:
                print("  (No sub-question results)")
            else:
                for hop_idx, sub_res in enumerate(hop_sub_results, start=1):
                    print(f"\n--- Hop Block {hop_idx} (3-hop joint block) ---")
                    analysis = sub_res.get("analysis", {})
                    if analysis:
                        # 3-hop 프롬프트에 맞게 필드가 있다면 출력
                        bc1 = analysis.get("bridge_connection1") or analysis.get("bridge_connection")
                        bc2 = analysis.get("bridge_connection2")
                        print(f"  Bridge connection 1: {bc1 or ''}")
                        if bc2 is not None:
                            print(f"  Bridge connection 2: {bc2}")
                        doc_a_seg = analysis.get("doc_a_seg", "")
                        doc_b_seg = analysis.get("doc_b_seg", "")
                        doc_c_seg = analysis.get("doc_c_seg", "")
                        if doc_a_seg:
                            print(f"  Doc A seg: {doc_a_seg[:200]}{'...' if len(doc_a_seg)>200 else ''}")
                        if doc_b_seg:
                            print(f"  Doc B seg: {doc_b_seg[:200]}{'...' if len(doc_b_seg)>200 else ''}")
                        if doc_c_seg:
                            print(f"  Doc C seg: {doc_c_seg[:200]}{'...' if len(doc_c_seg)>200 else ''}")
                        rp1 = analysis.get("reasoning_path1") or analysis.get("reasoning_path")
                        rp2 = analysis.get("reasoning_path2")
                        print(f"  Reasoning path 1: {rp1 or ''}")
                        if rp2 is not None:
                            print(f"  Reasoning path 2: {rp2}")

                    sub_qs = sub_res.get("sub_questions", [])
                    for i, sq in enumerate(sub_qs, start=1):
                        print(f"\n  Sub-question {i}: {sq.get('question','')}")
                        print(f"    Answer: {sq.get('answer','')}")
                        print(f"    Source: {sq.get('source','')}")
            
            # ============================
            # 4) 최종 multi-hop 질문
            # ============================
            print("\n=== Final Multi-hop Question ===")
            mh = result.get("multi_hop_question", {})
            print(f"Question: {mh.get('multi_hop_question','')}")
            print(f"Answer: {mh.get('answer','')}")
            print(f"Reasoning path: {mh.get('reasoning_path','')}")
            print(f"Sources: {mh.get('sources','')}")

            # ============================
            # 5) Polish 결과
            # ============================
            print("\n=== Polish Result ===")
            pr = result.get("polish_result")
            if pr:
                print(f"Status: {pr.get('status','UNKNOWN')}")
                if pr.get("status") in ["ADJUST", "REWORKED"]:
                    print(f"Refined question: {pr.get('refined_question','')}")
                    print(f"Refined reasoning: {pr.get('refined_reasoning_path','')}")
                    print(f"Refined answer: {pr.get('answer','')}")
            else:
                print("No polish or polish failed")
    else:
        # Batch generation mode
        print("=== Batch Question Generation Mode (3-hop) ===")
        batch_count = 5
        results = synthesizer.batch_generate(count=batch_count, save_all=True)
        print(f"\nBatch generation complete: success={len(results)}/{batch_count}")