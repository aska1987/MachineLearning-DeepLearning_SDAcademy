#export,import
--sqlplus에서
@C:\oraclexe\app\oracle\product\11.2.0\server\rdbms\admin\catexp.sql
--prompt에서(cmd)
exp system/manager tables=(hr.emp,hr.dept) grants=y indexes=y --지정 경로에 export 
-- grants: db객체에 대한 권한을 export할 것인지 여부
-- indexes: 테이블들에 대한 인덱스를 export 할 것인지 여부
-- tables: export할 테이블 

exp hr/hr file=hr.demp tables=emp,dept rows=y compress=y 
-- file: 파일명 및 파일 형태,
-- tables: export할 테이블 
-- rows: 행들이 export 될것인지 여부
-- compress: 세그먼트들을 단일확장영역들로 압출할지여부

exp hr/hr file=hr_all.dmp owner=hr grants=y rows=y compress=y --hr계정의 모든 데이터가 export
-- owner: export 될 사용자 목록

exp hr/hr full=y file=dba.dmp grants=y rows=y --전체 데이터베이스를 export
-- full: 전체 데이터베이스를 export 할지 여부

imp system/manager file=exdat.dmp fromuser=hr touser=system tables=(dept,emp) -- import
-- file: import될 파일명
-- fromuser: export dump파일로부터 읽혀져야 하는 객체들을 갖고있는 사용자 목록
--> system에서 확인

imp system/manager file=hr_all.dmp fromuser=hr touser=tiger --import\
--> tiger에서 확인

#1. vline/vline88 유저를 생성
create user vline
identified by vline88
default tablespace test temporary tablespace temp;
grant connect, resource to vline;
#2. 생성한 vline로 다음과 같은 속성으로 정의된 사원 테이블(s_custom)을 생성
create table s_custom(
id number(7) not null,
 name varchar2(50) not null,
 phone varchar2(25),
 address varchar2(50),
 zip_code varchar2(10),
 region_id number(7),
 comments varchar2(100),
 credit_rating varchar2(9));
#3. 위 테이블에 데이터 3건을 입력
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating)
values (112234,'하이','010-1111-2222','hi123@gmail.com','123412311',910102,null,'1등급');
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating) 
values (143114,'헬로우','010-6617-7222','hello993@naver.com','1099211',673712,'없음','3등급');
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating) 
values (182934,'가즈아','010-7817-3982','gauza11@daum.net','996123',88122,'가즈아','2등급');
#4. 데이터 수정
update s_custom
 set credit_rating='1등급'
 where id=143114;
#5. 데이터 삭제
delete from s_custom
 where id=143114;
#6. vline의 s_custom 테이블의 데이터를 덤프받아서 tiger로 import
exp vline/vline88 tables=s_custom grants=y indexes=y
imp system/manager file=expdat.dmp fromuser=vline touser=tiger tables=(s_custom);

#loader 실습
--테이블 만들기
create table dept1 as
 select * from dept
 where 1=0;
 --테이블에 데이터 로더
 --cmd창에서
 sqlldr userid=hr/hr control='C:\Loader실습\demo1.ctl' log=demo1.log --demo1.ctl에있는 내용을 로더
 select * from dept1; --데이터가 로더된 것을 확인
 
 --emp테이블에 데이터로더
 sqlldr userid=hr/hr control='C:\Loader실습\demo2.ctl' data='C:\Loader실습\demo2.dat' log=demo2.log
-- control: 컨트롤 파일이름
-- data: 입력 데이터 파일
-- log: 로그 파일 이름
select * from emp;


--테이블 생성
delete from emp;
create table emp_rating(
  empno number(4),
  leadership number(3),
  membershop number(3),
  english number(3),
  computing number(3));
--로더
sqlldr userid=hr/hr control=demo5.ctl log=demo2.log
--확인
select * from emp;
select * from emp_rating;

