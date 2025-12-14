import os
import json
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flashrag.generator import OpenaiGenerator
from flashrag.config import Config
from hopweaver.components.utils.prompts import SUB_QUESTION_GENERATION_PROMPT, MULTI_HOP_QUESTION_SYNTHESIS_PROMPT, SUB_QUESTION_GENERATION_3HOP_PROMPT, HOP3_QUESTION_SYNTHESIS_PROMPT

class QuestionGenerator:
    """Multi-hop question generator"""
    def __init__(self, config):
        self.config = config
        
        # Set model configuration
        if "question_generator_model" in self.config:
            self.config["generator_model"] = self.config["question_generator_model"]
        elif "generator_model" not in self.config:
            self.config["generator_model"] = "gpt-4o"
            
        # Set maximum token count
        if "generation_params" not in self.config:
            self.config["generation_params"] = {}
        self.config["generation_params"]["max_completion_tokens"] = 4096
        
        # Initialize generator
        self.generator = self._initialize_model()
        
    def _initialize_model(self):
        """Initialize OpenAI generator"""
        # Use FlashRAG's OpenaiGenerator
        return OpenaiGenerator(self.config)
        
    def generate_sub_questions(self, bridge_entity, entity_type, doc_a_segments, doc_b_document, max_retry=3):
        """Generate multi-hop sub-questions"""
        # Parameter validation
        if not bridge_entity or not doc_a_segments or not doc_b_document:
            print("Warning: Missing necessary input parameters")
            return None
            
        # Prepare prompt template
        prompt = SUB_QUESTION_GENERATION_PROMPT.format(
            bridge_entity=bridge_entity,
            entity_type=entity_type if entity_type else "entity",
            doc_a_segments=doc_a_segments,
            doc_b_document=doc_b_document
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Implement retry mechanism
        for attempt in range(max_retry):
            try:
                if attempt > 0:
                    print(f"Retrying attempt {attempt + 1}...")
                
                # Call generator
                response = self.generator.generate([messages])
                
                if not response or response[0] is None:
                    print("Empty response")
                    continue
                
                # Check for invalid bridge connection - before parsing the response
                if response[0].replace("<|COMPLETE|>", "").strip().startswith("INVALID_BRIDGE_CONNECTION"):
                    # If an invalid bridge connection is detected, return None directly, do not retry
                    reason = response[0].split("\n")[1] if len(response[0].split("\n")) > 1 else "No reason provided"
                    print("Invalid bridge connection:", reason)
                    return None
                print("Response:", response[0])
                # Parse response
                result = self._parse_sub_questions_response(response[0])
                print("Parsed result:", result)
                # Check if the result is complete
                if result and "analysis" in result and "sub_questions" in result:
                    # Check if Document B paragraph contains at least 20 words
                    if "analysis" in result and "doc_b_seg" in result["analysis"]:
                        doc_b_seg_words = len(result["analysis"]["doc_b_seg"].split())
                        if doc_b_seg_words < 20:
                            print(f"Document B paragraph has only {doc_b_seg_words} words, less than the required 20 words, trying to regenerate...")
                            continue
                    
                    if len(result["sub_questions"]) == 2:
                        return result
                    
            except Exception as e:
                print(f"Error generating sub-questions (attempt {attempt+1}/{max_retry}): {str(e)}")
                
        return None
    def generate_3_sub_questions(
        self,
        bridge_entity1,
        entity_type1,
        bridge_entity2,
        entity_type2,
        doc_a_segments,
        doc_b_document,
        doc_c_document,
        max_retry=3,
    ):
        """
        3-hop (문서 A,B,C + bridge_entity1, bridge_entity2)용 서브질문 생성 함수.

        Inputs:
            bridge_entity1: str, A→B를 잇는 브리지 엔티티 이름
            entity_type1  : str, bridge_entity1 타입
            bridge_entity2: str, B→C를 잇는 브리지 엔티티 이름
            entity_type2  : str, bridge_entity2 타입
            doc_a_segments: str, Document A 관련 segment (이미 추출된 부분)
            doc_b_document: str, Document B 전체(or 주요) 텍스트
            doc_c_document: str, Document C 전체(or 주요) 텍스트
        Returns:
            {
                "analysis": {
                    "bridge_connection1",
                    "bridge_connection2",
                    "doc_a_seg",
                    "doc_b_seg",
                    "doc_c_seg",
                    "reasoning_path1",
                    "reasoning_path2",
                },
                "sub_questions": [
                    {"question", "answer", "source"},  # A
                    {"question", "answer", "source"},  # B
                    {"question", "answer", "source"},  # C
                ]
            }
            or None (실패 / INVALID_BRIDGE_CONNECTION)
        """
        # 기본 입력 체크
        if (
            not bridge_entity1
            or not bridge_entity2
            or not doc_a_segments
            or not doc_b_document
            or not doc_c_document
        ):
            print("Warning: Missing necessary input parameters for 3-hop sub-question generation")
            return None

        # 프롬프트 구성
        prompt = SUB_QUESTION_GENERATION_3HOP_PROMPT.format(
            bridge_entity1=bridge_entity1,
            bridge_entity2=bridge_entity2,
            entity_type1=entity_type1 if entity_type1 else "entity",
            entity_type2=entity_type2 if entity_type2 else "entity",
            doc_a_segments=doc_a_segments,
            doc_b_document=doc_b_document,
            doc_c_document=doc_c_document,
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retry):
            try:
                if attempt > 0:
                    print(f"[3-hop] Retrying attempt {attempt+1}/{max_retry}...")

                response = self.generator.generate([messages])
                if not response or response[0] is None:
                    print("[3-hop] Empty response")
                    continue

                raw = response[0]
                cleaned = raw.replace("<|COMPLETE|>", "").strip()

                # INVALID_BRIDGE_CONNECTION 처리
                if cleaned.startswith("INVALID_BRIDGE_CONNECTION"):
                    reason = ""
                    parts = cleaned.split("\n", 1)
                    if len(parts) > 1:
                        reason = parts[1]
                    print("[3-hop] Invalid bridge connection:", reason)
                    return None

                print("[3-hop] Raw response:", raw)

                # 3-hop 전용 파서로 파싱
                result = self._parse_3hop_sub_questions_response(raw)
                print("[3-hop] Parsed result:", result)

                if (
                    result
                    and "analysis" in result
                    and "sub_questions" in result
                    and len(result["sub_questions"]) == 3
                ):
                    # Document B, C segment 최소 길이 체크 (옵션)
                    analysis = result.get("analysis", {})
                    if "doc_b_seg" in analysis:
                        b_words = len(analysis["doc_b_seg"].split())
                        if b_words < 20:
                            print(f"[3-hop] Document B segment too short ({b_words} words), retrying...")
                            continue
                    if "doc_c_seg" in analysis:
                        c_words = len(analysis["doc_c_seg"].split())
                        if c_words < 20:
                            print(f"[3-hop] Document C segment too short ({c_words} words), retrying...")
                            continue

                    return result

            except Exception as e:
                print(f"[3-hop] Error generating sub-questions (attempt {attempt+1}/{max_retry}): {str(e)}")

        return None
    
    def _parse_sub_questions_response(self, response):
        """Parse sub-question response, format refers to SUB_QUESTION_GENERATION_PROMPT"""
        try:
            # Remove possible completion mark
            response = response.replace("<|COMPLETE|>", "").strip()
            
            # Check if it is an invalid bridge connection response
            if response.startswith("INVALID_BRIDGE_CONNECTION"):
                print("Invalid bridge connection:", response.split("\n")[1] if len(response.split("\n")) > 1 else "No reason provided")
                return None
                
            # Separate analysis and sub-question parts
            if "SUB-QUESTIONS:" not in response:
                return None
                
            analysis_part = response.split("SUB-QUESTIONS:")[0].strip()
            sub_questions_part = "SUB-QUESTIONS:" + response.split("SUB-QUESTIONS:")[1].strip()
            
            # Parse analysis part
            analysis = {}
            
            # Use a more accurate method to extract the content of each part
            # Define all possible paragraph markers
            section_markers = [
                "Bridge connection:",
                "Document A segments:",
                "Document B segments:",
                "Reasoning path:",
                "SUB-QUESTIONS:"
            ]
            
            # Split text by these markers
            sections = {}
            current_marker = None
            current_content = []
            
            lines = analysis_part.split("\n")
            for line in lines:
                # Check if the current line is the start of a new paragraph
                found_marker = False
                for marker in section_markers:
                    if line.startswith(marker):
                        # Save the content of the previous paragraph
                        if current_marker:
                            sections[current_marker] = "\n".join(current_content)
                        
                        # Start a new paragraph
                        current_marker = marker
                        # Extract content from the same line
                        content_in_line = line.replace(marker, "").strip()
                        current_content = [content_in_line] if content_in_line else []
                        found_marker = True
                        break
                
                # If not the start of a new paragraph, add to the current paragraph
                if not found_marker and current_marker and line.strip():
                    current_content.append(line.strip())
            
            # Add the last paragraph
            if current_marker and current_content:
                sections[current_marker] = "\n".join(current_content)
            
            # Map markers to analysis fields
            if "Bridge connection:" in sections:
                analysis["bridge_connection"] = sections["Bridge connection:"]
            if "Document A segments:" in sections:
                analysis["doc_a_seg"] = sections["Document A segments:"]
            if "Document B segments:" in sections:
                analysis["doc_b_seg"] = sections["Document B segments:"]
            if "Reasoning path:" in sections:
                analysis["reasoning_path"] = sections["Reasoning path:"]
                
            # Output debug information
            print(f"Extracted Document A paragraph: {len(analysis.get('doc_a_seg', '').split()) if 'doc_a_seg' in analysis else 0} words")
            print(f"Extracted Document B paragraph: {len(analysis.get('doc_b_seg', '').split()) if 'doc_b_seg' in analysis else 0} words")
            
            # Parse sub-question part
            sub_questions = []
            for i in range(1, 3):  # Process two sub-questions
                question_marker = f"Sub-question {i}:"
                answer_marker = f"Answer {i}:"
                next_marker = f"Sub-question {i+1}:" if i == 1 else ""  # For the second question, there is no next marker
                
                if question_marker in sub_questions_part and answer_marker in sub_questions_part:
                    question = sub_questions_part.split(question_marker)[1].split(answer_marker)[0].strip()
                    
                    # Handle splitting differently based on whether there is a next question
                    if i == 1 and next_marker in sub_questions_part:
                        answer = sub_questions_part.split(answer_marker)[1].split(next_marker)[0].strip()
                    else:  # Last question or only one question
                        answer = sub_questions_part.split(answer_marker)[1].strip()
                    
                    sub_questions.append({
                        "question": question,
                        "answer": answer,
                        "source": f"Document {'A' if i == 1 else 'B'}"
                    })
            
            return {"analysis": analysis, "sub_questions": sub_questions}
            
        except Exception as e:
            print(f"Error parsing sub-question response: {str(e)}")
            first_line = response.split("\n")[0] if response else "(Empty response)"
            print(f"Response content first line: {first_line}")
            return None
    
    def _parse_3hop_sub_questions_response(self, response):
        """
        SUB_QUESTION_GENERATION_3HOP_PROMPT 포맷에 맞는 3-hop 서브질문 응답 파서.

        기대 포맷:
        - INVALID_BRIDGE_CONNECTION ...  (이 경우 None 리턴)
        - 아니면:

          ANALYSIS:
          Bridge connection 1: ...
          Bridge connection 2: ...
          Document A segments: ...
          Document B segments: ...
          Document C segments: ...
          Reasoning path 1: ...
          Reasoning path 2: ...

          SUB-QUESTIONS:
          Sub-question 1: ...
          Answer 1: ...
          Sub-question 2: ...
          Answer 2: ...
          Sub-question 3: ...
          Answer 3: ...
        """
        try:
            response = response.replace("<|COMPLETE|>", "").strip()

            # INVALID_BRIDGE_CONNECTION 처리 (안전망)
            if response.startswith("INVALID_BRIDGE_CONNECTION"):
                print(
                    "[3-hop] Invalid bridge connection:",
                    response.split("\n")[1] if len(response.split("\n")) > 1 else "No reason provided",
                )
                return None

            if "SUB-QUESTIONS:" not in response:
                print("[3-hop] 'SUB-QUESTIONS:' not found in response")
                return None

            analysis_part, sub_questions_part = response.split("SUB-QUESTIONS:", 1)
            analysis_part = analysis_part.strip()
            sub_questions_part = "SUB-QUESTIONS:" + sub_questions_part.strip()

            # ---- ANALYSIS 파싱 ----
            analysis = {}
            section_markers = [
                "Bridge connection 1:",
                "Bridge connection 2:",
                "Document A segments:",
                "Document B segments:",
                "Document C segments:",
                "Reasoning path 1:",
                "Reasoning path 2:",
            ]

            sections = {}
            current_marker = None
            current_content = []

            lines = analysis_part.split("\n")
            for line in lines:
                stripped = line.rstrip()
                if not stripped:
                    continue

                found_marker = False
                for marker in section_markers:
                    if stripped.startswith(marker):
                        # 이전 섹션 저장
                        if current_marker is not None:
                            sections[current_marker] = "\n".join(current_content).strip()

                        current_marker = marker
                        # 같은 줄에 내용이 있으면 바로 추출
                        content_in_line = stripped[len(marker):].strip()
                        current_content = [content_in_line] if content_in_line else []
                        found_marker = True
                        break

                if not found_marker and current_marker is not None:
                    current_content.append(stripped)

            if current_marker is not None and current_content:
                sections[current_marker] = "\n".join(current_content).strip()

            # sections → analysis 매핑
            if "Bridge connection 1:" in sections:
                analysis["bridge_connection1"] = sections["Bridge connection 1:"]
            if "Bridge connection 2:" in sections:
                analysis["bridge_connection2"] = sections["Bridge connection 2:"]
            if "Document A segments:" in sections:
                analysis["doc_a_seg"] = sections["Document A segments:"]
            if "Document B segments:" in sections:
                analysis["doc_b_seg"] = sections["Document B segments:"]
            if "Document C segments:" in sections:
                analysis["doc_c_seg"] = sections["Document C segments:"]
            if "Reasoning path 1:" in sections:
                analysis["reasoning_path1"] = sections["Reasoning path 1:"]
            if "Reasoning path 2:" in sections:
                analysis["reasoning_path2"] = sections["Reasoning path 2:"]

            print(
                "[3-hop] Extracted segments word counts:",
                "A =", len(analysis.get("doc_a_seg", "").split()),
                "B =", len(analysis.get("doc_b_seg", "").split()),
                "C =", len(analysis.get("doc_c_seg", "").split()),
            )

            # ---- SUB-QUESTIONS 파싱 ----
            sub_questions = []
            for i in range(1, 4):  # 1,2,3
                q_marker = f"Sub-question {i}:"
                a_marker = f"Answer {i}:"
                next_q_marker = f"Sub-question {i+1}:" if i < 3 else None

                if q_marker not in sub_questions_part or a_marker not in sub_questions_part:
                    continue

                # 질문 추출
                after_q = sub_questions_part.split(q_marker, 1)[1]
                if a_marker in after_q:
                    question = after_q.split(a_marker, 1)[0].strip()
                else:
                    question = after_q.strip()

                # 답 추출
                after_a = sub_questions_part.split(a_marker, 1)[1]
                if next_q_marker and next_q_marker in after_a:
                    answer = after_a.split(next_q_marker, 1)[0].strip()
                else:
                    answer = after_a.strip()

                # source 태그: 1→A, 2→B, 3→C
                if i == 1:
                    source = "Document A"
                elif i == 2:
                    source = "Document B"
                else:
                    source = "Document C"

                sub_questions.append(
                    {
                        "question": question,
                        "answer": answer,
                        "source": source,
                    }
                )

            if not sub_questions:
                print("[3-hop] No sub-questions parsed")
                return None

            return {
                "analysis": analysis,
                "sub_questions": sub_questions,
            }

        except Exception as e:
            print(f"[3-hop] Error parsing 3-hop sub-question response: {str(e)}")
            first_line = response.split("\n")[0] if response else "(Empty response)"
            print(f"[3-hop] Response content first line: {first_line}")
            return None

    def synthesize_3_hop_question(self, hop_sub_results, documents=None, bridge_entities=None, max_retry=1):
        """
        3-hop 전용 multi-hop 질문 합성 함수.

        기대 입력 형식:
            hop_sub_results: 길이 1인 리스트
              hop_sub_results[0] = {
                  "analysis": {
                      "bridge_connection1", "bridge_connection2",
                      "doc_a_seg", "doc_b_seg", "doc_c_seg",
                      "reasoning_path1", "reasoning_path2",
                      ...
                  },
                  "sub_questions": [
                      {"question", "answer", "source"},  # Sub-question 1 (Document A)
                      {"question", "answer", "source"},  # Sub-question 2 (Document B)
                      {"question", "answer", "source"},  # Sub-question 3 (Document C)
                  ],
              }

        documents, bridge_entities 인자는 현재 프롬프트에는 직접 사용하지 않지만,
        상위 호출부와의 인터페이스를 맞추기 위해 남겨둔다.
        """
        # ------------------------------
        # 입력 검증
        # ------------------------------
        if not hop_sub_results or len(hop_sub_results) != 1:
            print("[3-hop] synthesize_3_hop_question expects hop_sub_results to be a single aggregated 3-hop block.")
            print(f"[3-hop] got len(hop_sub_results) = {len(hop_sub_results) if hop_sub_results is not None else 'None'}")
            return None

        block = hop_sub_results[0]
        analysis = block.get("analysis", {}) or {}
        sub_qs = block.get("sub_questions", []) or []

        if len(sub_qs) != 3:
            print(f"[3-hop] Expected exactly 3 sub-questions, but got {len(sub_qs)}")
            return None

        # ------------------------------
        # HOP SUB-RESULT 문자열 구성 (프롬프트의 {hop_sub_results} 자리에 들어갈 텍스트)
        # ------------------------------
        lines = []
        lines.append("ANALYSIS:")

        # bridge connections
        if "bridge_connection1" in analysis:
            lines.append(f"Bridge connection 1: {analysis.get('bridge_connection1', '')}")
        if "bridge_connection2" in analysis:
            lines.append(f"Bridge connection 2: {analysis.get('bridge_connection2', '')}")

        # segments
        lines.append(f"Document A segments: {analysis.get('doc_a_seg', '')}")
        lines.append(f"Document B segments: {analysis.get('doc_b_seg', '')}")
        lines.append(f"Document C segments: {analysis.get('doc_c_seg', '')}")

        # reasoning paths
        if "reasoning_path1" in analysis:
            lines.append(f"Reasoning path 1: {analysis.get('reasoning_path1', '')}")
        if "reasoning_path2" in analysis:
            lines.append(f"Reasoning path 2: {analysis.get('reasoning_path2', '')}")

        lines.append("")
        lines.append("SUB-QUESTIONS:")

        for i, q in enumerate(sub_qs, start=1):
            q_text = q.get("question", "")
            a_text = q.get("answer", "")
            src = q.get("source", "")
            lines.append(f"Sub-question {i}: {q_text}")
            lines.append(f"Answer {i}: {a_text}")
            if src:
                lines.append(f"Source: {src}")
            lines.append("")

        hop_sub_results_str = "\n".join(lines).strip()

        # ------------------------------
        # 프롬프트 구성 (3-hop 전용 MULTI_HOP_QUESTION_SYNTHESIS_PROMPT 사용)
        # ------------------------------
        prompt = HOP3_QUESTION_SYNTHESIS_PROMPT.format(
            hop_sub_results=hop_sub_results_str
        )

        messages = [{"role": "user", "content": prompt}]

        # ------------------------------
        # LLM 호출 + 파싱
        # ------------------------------
        for attempt in range(max_retry):
            try:
                if attempt > 0:
                    print(f"[3-hop] Retrying 3-hop synthesis (attempt {attempt+1}/{max_retry})...")

                response = self.generator.generate([messages])
                if not response or response[0] is None:
                    print("[3-hop] Empty response from generator in 3-hop synthesis")
                    continue

                raw = response[0]
                cleaned = raw.replace("<|COMPLETE|>", "").strip()

                # NONE 패턴 처리
                if cleaned.startswith("NONE"):
                    parts = cleaned.split("\n", 1)
                    reason_line = parts[1] if len(parts) > 1 else ""
                    print("[3-hop] 3-hop synthesis returned NONE:", reason_line)
                    return None

                # 공통 파서 재사용 (출력 포맷은 2-hop 때와 동일 헤더 사용)
                result = self._parse_multi_hop_synthesis_response(raw)
                if result and "multi_hop_question" in result and "answer" in result:
                    return result

            except Exception as e:
                print(f"[3-hop] Error in 3-hop synthesis (attempt {attempt+1}/{max_retry}): {str(e)}")

        return None

    def synthesize_multi_hop_question(self, sub_questions_result, max_retry=1):
        """Synthesize multi-hop question
        
        Args:
            sub_questions_result (dict): Sub-question generation result
            max_retry (int): Maximum number of retries (default is 1, no retry)
            
        Returns:
            dict: Dictionary containing multi-hop question, answer, reasoning path, and source
        """
        # Check if input is valid
        if not sub_questions_result or "analysis" not in sub_questions_result or "sub_questions" not in sub_questions_result:
            print("Warning: Insufficient input parameters")
            return None
            
        # Ensure there are two complete sub-questions
        if len(sub_questions_result["sub_questions"]) != 2:
            print(f"Warning: Incorrect number of sub-questions (expected: 2, actual: {len(sub_questions_result['sub_questions'])})")
            return None
        
        # Format analysis and sub-questions as readable strings
        analysis_str = ""
        for key, value in sub_questions_result["analysis"].items():
            analysis_str += f"{key.replace('_', ' ').title()}: {value}\n"
        
        sub_questions_str = ""
        for i, q in enumerate(sub_questions_result["sub_questions"]):
            sub_questions_str += f"Sub-question {i+1}: {q.get('question', '')}\n"
            sub_questions_str += f"Answer {i+1}: {q.get('answer', '')}\n"
            sub_questions_str += f"Source: {q.get('source', '')}\n\n"
        
        # Prepare prompt template
        prompt = MULTI_HOP_QUESTION_SYNTHESIS_PROMPT.format(
            analysis=analysis_str,
            sub_questions=sub_questions_str
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Call generator
            response = self.generator.generate([messages])
            
            if not response or response[0] is None:
                return None
            
            # Parse response
            result = self._parse_multi_hop_synthesis_response(response[0])
            
            # Check if result is complete
            if result and "multi_hop_question" in result and "answer" in result:
                return result
                
        except Exception as e:
            print(f"Error synthesizing multi-hop question: {str(e)}")
            
        return None
    
    def _parse_multi_hop_synthesis_response(self, response):
        """Parse multi-hop question response, format refers to MULTI_HOP_QUESTION_SYNTHESIS_PROMPT"""
        try:
            # Remove possible completion mark
            response = response.replace("<|COMPLETE|>", "").strip()
            
            # Check if it contains a JSON code block
            if response.startswith("NONE"):
                print("Failed to create a valid multi-hop question:", response.split("\n")[1] if len(response.split("\n")) > 1 else "No reason provided")
                return None
            
            # Check basic format
            if "MULTI-HOP QUESTION:" not in response:
                return None
                
            # Parse each part
            result = {}
            
            # Extract multi-hop question
            question_part = response.split("MULTI-HOP QUESTION:")[1].split("ANSWER:")[0].strip()
            result["multi_hop_question"] = question_part
            
            # Extract answer
            answer_part = response.split("ANSWER:")[1].split("REASONING PATH:")[0].strip()
            result["answer"] = answer_part
            
            # Extract reasoning path
            reasoning_part = response.split("REASONING PATH:")[1].split("SOURCES:")[0].strip()
            result["reasoning_path"] = reasoning_part
            
            # Extract source
            sources_part = response.split("SOURCES:")[1].strip()
            result["sources"] = sources_part
            
            return result
            
        except Exception as e:
            print(f"Error parsing multi-hop question response: {str(e)}")
            first_line = response.split("\n")[0] if response else "(Empty response)"
            print(f"Response content first line: {first_line}")
            return None

if __name__ == "__main__":
    # Test code
    
    # Configuration
    config = Config("./config_lib/example_config.yaml", {})
    
    # Create question generator
    question_generator = QuestionGenerator(config)
    
    # Test input
    bridge_entity = "Charlie Wilson" # Example bridge entity
    entity_type = "Person" # Example entity type
    doc_a_segments = """During much of the 1980s a unique and unusual relationship evolved between Congress and the CIA in the person of Texas congressman Charlie Wilson from Texas's 2nd congressional district. Using his position on various House appropriations committees, and in partnership with CIA agent Gust Avrakotos, Wilson was able to increase CIA's funding the Afghan Mujahideen to several hundred million dollars a year during the Soviet Afghan war. Author George Crile would describe Wilson as eventually becoming the \\\"Agency's station chief on the Hill\\\". Charlie Wilson eventually got a position on the Intelligence Committee and was supposed to be overseeing the CIA.
    """
    
    doc_b_document = """Charlie Wilson's War (film)\nCharlie Wilson's War (film) Charlie Wilson's War is a 2007 American biographical comedy-drama film, based on the story of U.S. Congressman Charlie Wilson and CIA operative Gust Avrakotos, whose efforts led to Operation Cyclone, a program to organize and support the Afghan mujahideen during the Soviet–Afghan War. The film was directed by Mike Nichols (his final film) and written by Aaron Sorkin, who adapted George Crile III's 2003 book \"\". Tom Hanks, Julia Roberts, and Philip Seymour Hoffman starred, with Amy Adams and Ned Beatty in supporting roles. It was nominated for five Golden Globe Awards, including Best Motion Picture – Musical or Comedy, but did not win in any category. Hoffman was nominated for an Academy Award for Best Supporting Actor. In 1980, Congressman Charlie Wilson is more interested in partying than legislating, frequently throwing huge galas and staffing his congressional office with young, attractive women. His social life eventually brings about a federal investigation into allegations of his cocaine use, conducted by federal prosecutor Rudy Giuliani as part of a larger investigation into congressional misconduct. The investigation results in no charge against Charlie. A friend and romantic interest, Joanne Herring, encourages Charlie to do more to help the Afghan people, and persuades Charlie to visit the Pakistani leadership. The Pakistanis complain about the inadequate support of the U.S. to oppose the Soviet Union, and they insist that Charlie visit a major Pakistan-based Afghan refugee camp. Charlie is deeply moved by their misery and determination to fight, but is frustrated by the regional CIA personnel's insistence on a low key approach against the Soviet occupation of Afghanistan. Charlie returns home to lead an effort to substantially increase funding to the mujahideen. As part of this effort, Charlie befriends maverick CIA operative Gust Avrakotos and his understaffed Afghanistan group to find a better strategy, especially including a means to counter the Soviets' formidable Mi-24 helicopter gunship. This group was composed in part of members of the CIA's Special Activities Division, including a young paramilitary officer named Michael Vickers. As a result, Charlie's deft political bargaining for the necessary funding and Avrakotos' careful planning using those resources, such as supplying the guerrillas with FIM-92 Stinger missile launchers, turns the Soviet occupation into a deadly quagmire with their heavy fighting vehicles being destroyed at a crippling rate. Charlie enlists the support of Israel and Egypt for Soviet weapons and consumables, and Pakistan for distribution of arms. The CIA's anti-communism budget evolves from $5 million to over $500 million (with the same amount matched by Saudi Arabia), startling several congressmen. This effort by Charlie ultimately evolves into a major portion of the U.S. foreign policy known as the Reagan Doctrine, under which the U.S. expanded assistance beyond just the mujahideen and began also supporting other anti-communist resistance movements around the world. Charlie states that senior Pentagon official Michael Pillsbury persuaded President Ronald Reagan to provide the Stingers to the Afghans. Charlie follows Gust's guidance to seek support for post-Soviet occupation Afghanistan, but finds no enthusiasm in the government for even the modest measures he proposes. In the end, Charlie receives a major commendation for his support of the U.S. clandestine services, but his pride is tempered by his fears of the blowback his secret efforts could yield in the future and the implications of U.S. disengagement from Afghanistan. The film was originally set for release on December 25, 2007; but on November 30, the timetable was moved up to December 21. In its opening weekend, the film grossed $9.6 million in 2,575 theaters in the United States and Canada, ranking #4 at the box office. It grossed a total of $119 million worldwide—$66.7 million in the United States and Canada and $52.3 million in other territories. On review aggregator Rotten Tomatoes, the film has an approval rating of 82% based on 196 reviews, with an average rating of 6.8/10. The site's critical consensus reads, \"\"Charlie Wilson's War\" manages to entertain and inform audiences, thanks to its witty script and talented cast of power players.\" Metacritic reported the film had an average score of 69 out of 100, based on 39 critics, indicating \"generally favorable reviews\". Audiences polled by CinemaScore gave the film an average grade of \"A–\" on an A+ to F scale. Some Reagan-era officials, including former Under Secretary of Defense Fred Ikle, have criticized some elements of the film. \"The Washington Times\" reported claims that the film wrongly promotes the notion that the CIA-led operation funded Osama bin Laden and ultimately produced the September 11 attacks. Other Reagan-era officials, however, have been more supportive of the film. Michael Johns, the former foreign policy analyst at The Heritage Foundation and White House speechwriter to President George H. W. Bush, praised the film as \"the first mass-appeal effort to reflect the most important lesson of America's Cold War victory: that the Reagan-led effort to support freedom fighters resisting Soviet oppression led successfully to the first major military defeat of the Soviet Union... Sending the Red Army packing from Afghanistan proved one of the single most important contributing factors in one of history's most profoundly positive and important developments.\" In February 2008, it was revealed that the film would not play in Russian theaters. The rights for the film were bought by Universal Pictures International (UPI) Russia. It was speculated that the film would not appear because of a certain point of view that depicted the Soviet Union unfavorably. UPI Russia head Yevgeny Beginin denied that, saying, \"We simply decided that the film would not make a profit.\" Reaction from Russian bloggers was also negative. One wrote: \"The whole film shows Russians, or rather Soviets, as brutal killers.\" While the film depicts Wilson as an immediate advocate for supplying the mujahideen with Stinger missiles, a former Reagan administration official recalls that he and Wilson, while advocates for the mujahideen, were actually initially \"lukewarm\" on the idea of supplying these missiles. Their opinion changed when they discovered that rebels were successful in downing Soviet gunships with them. As such, they were actually not supplied until the second Reagan administration term, in 1987, and their provision was advocated mostly by Reagan defense officials and influential conservatives. The film's happy ending came about because Tom Hanks \"just can't deal with this 9/11 thing,\" according to Melissa Roddy, a Los Angeles film maker with inside information from the production. Citing the original screenplay, which was very different from the final product, in \"\" Matthew Alford wrote that the film gave up \"the chance to produce what at least had the potential to be the Dr. Strangelove of our generation\". In his 2011 book \"Afgantsy,\" former British ambassador to Russia Rodric Braithwaite describes the film as \"amusing but has only an intermittent connection with historical reality.\" The film depicts the concern expressed by Charlie and Gust that Afghanistan was being neglected in the 1990s, following the Soviet withdrawal. In one of the film's final scenes, Gust dampens Charlie's enthusiasm over the Soviet withdrawal from Afghanistan, saying \"I'm about to give you an NIE (National Intelligence Estimate) that shows the crazies are rolling into Kandahar.\" George Crile III, author of the book on which the film is based, wrote that the mujahideen's victory in Afghanistan ultimately opened a power vacuum for bin Laden: \"By the end of 1993, in Afghanistan itself there were no roads, no schools, just a destroyed country—and the United States was washing its hands of any responsibility. It was in this vacuum that the Taliban and Osama bin Laden would emerge as the dominant players. It is ironic that a man who had almost nothing to do with the victory over the Red Army, Osama bin Laden, would come to personify the power of the jihad.\" In 2008, Canadian journalist and politician Arthur Kent sued the makers of the film, claiming that they had used material he produced in the 1980s without obtaining the proper authorization. On September 19, 2008, Kent announced that he had reached a settlement with the film's producers and distributors, and that he was \"very pleased\" with the terms of the settlement, which remain confidential. The film was released on DVD April 22, 2008; a DVD version and a HD DVD/DVD combo version are available. The extras include a making of featurette and a \"Who is Charlie Wilson?\" featurette, which profiles the real Charlie Wilson and features interviews with him and with Tom Hanks, Joanne Herring, Aaron Sorkin, and Mike Nichols. The HD DVD/DVD combo version also includes additional exclusive content.
    """
    
    print("\n=== Step 1: Generate multi-hop sub-questions ===")
    # Generate sub-questions
    sub_questions_result = question_generator.generate_sub_questions(
        bridge_entity=bridge_entity,
        entity_type=entity_type,
        doc_a_segments=doc_a_segments,
        doc_b_document=doc_b_document
    )
    
    # Output sub-question results
    print("\nGenerated multi-hop sub-questions:")
    print(json.dumps(sub_questions_result, ensure_ascii=False, indent=2))
    
    # If sub-question generation is successful, proceed to synthesize multi-hop question
    if sub_questions_result:
        print("\n=== Step 2: Synthesize multi-hop question ===")
        # Synthesize multi-hop question
        multi_hop_result = question_generator.synthesize_multi_hop_question(sub_questions_result)
        
        # Output synthesis result
        print("\nSynthesized multi-hop question:")
        print(json.dumps(multi_hop_result, ensure_ascii=False, indent=2))
    else:
        print("\nSub-question generation failed, cannot proceed to synthesize multi-hop question")
