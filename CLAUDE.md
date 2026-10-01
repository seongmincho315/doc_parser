# doc-parer(전처리기)

- Fork 과정: docling -> https://github.com/genonai/doc_parser -> https://github.com/seongmincho315/doc_parser
- https://github.com/genonai/GenOS 
- 절대 하지 말아야 할것:

## TODO
0. 주기적으로 
   0.1. 현재 레포의 브랜치를 https://github.com/genonai/doc_parser 의 develop브랜치로 리베이스
      - 무시해도 될 파일: 
         - ./CLADUE.md
         - ./READEME.md
   0.2. mkdocs로 작성해서 우리 전처리기 사용하기 쉽게
1. TableFormer 파드 따로 띄워서 서빙(Paddle 처럼)
   - 현재 구조 단점:
      - 인스턴스마다 TableFormer를 띄워서 자원낭비
      - CPU로 전처리기 서빙시 -> 느려짐
      - GPU로 전처리기 서빙시 -> 각 페이지의 테이블을 한번에 테이블 포머로 보내기 때문에 OOM 될 리스크
2. dots-mocr triton으로 서빙
3. 다른 VLM-OCR 연동
   - FireRed
   - Abot
4. 전처리기 이미지에서 torch 빼는게 가능할지??
   - unstructured -> 파드
   - reading-order -> 파이썬으로 옮겨놓기
   - easy-ocr -> 안쓰니까 지움
5. 여러 전처리기 통합

