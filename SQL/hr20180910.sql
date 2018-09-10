#인덱스 정보 조회
select index_name,uniqueness
from user_indexes
where table_name='STUDENT';

select index_name,column_name
from user_ind_columns
where table_name='STUDENT';
#인덱스 삭제
drop index fidx_stud_no_name;
#인덱스 재구성
alter index stud_no_pk rebuild;

#1. EMPNO,ENAME,DEPTNO 열만을 포함하는 EMP 테이블의 구조를 기초로 EMPLOYEE2 테이블 생성, 새 테이블에서 ID,LAST_NAME,DEPT_ID 로 열 이름 지정
select * from emp;
create table EMPLOYEE2 as
select empno,ename,deptno
from emp;
select * from employee2;
alter table employee2
rename column empno to id;
alter table employee2
rename column ename to last_name;
alter table employee2
rename column deptno to dept_id;

#2. 생성한 EMPLOYEE2 테이블에서 LAST_NAME 필드를 10->30 으로 수정
alter table employee2
modify(LAST_NAME varchar2(30));
desc employee2;
#3-1. 사원 테이블의 DEPTNO와ENAME로 결합인덱스를 생성하라. 결합 인덱스의 이름은 fidx_emp_no_ename(단, DEPTNO 칼럼을 내림차순으로 ENAME컬럼은 오름차순으로 생성
create index fidx_emp_no_ename
on emp(DEPTNO asc,ENAME desc);
#3-2. 아래 질의의 인덱스의 실행경로를 확인하세요
SQLPLUS로
conn hr/hr
set autotrace on 
select empno,ename,deptno
from emp
where ename='SMITH'
and deptno=20;
#3-3. 인덱스를 재구성하세요.
alter index fidx_emp_no_ename rebuild;
#3-4. fidx_emp_no_ename 인덱스를 삭제하고 아래 질의의 인덱스의 실행경로를 확인하세요.
drop index fidx_emp_no_ename;
select empno,ename,deptno
from emp
where ename='SMITH'
and deptno=20;

#뷰생성
create view v_stud_dept102 as
select studno,name,deptno
from student
where deptno=102;

select * from v_stud_dept102;

crate view v_stud_dept102_2 as
select s.studno,s.name,s.grade,d.dname
from student s,department d
where s.deptno=d.deptno and s.deptno=102;

create view v_prof_avg_sal
as select deptno,sum(sal) sum_sal,avg(sal) avg_sal
from professor
group by deptno;

#인라인 뷰
select dname,avg_height,avg_weight
from (select deptno,avg(height) avg_height,avg(weight) avg_weight
from student
group by deptno) s,department d
where s.deptno=d.deptno;

#뷰 조회
select view_name,text
from user_views;

#뷰 변경
create or replace view v_stud_dept102
as select studno,name,deptno,grade
from student
where deptno=101;
#뷰 삭제
drop view v_stud_dept102;
drop view v_stud_dept102_2;

#단순 뷰
#1. PROFNO,NAME,POSITION,DEPTNO 열만 포함하는 PROFSSOR 테이블의 구조를 기초로 PROFESSOR2뷰를 작성
create view professor2
as select profno,name,position,deptno
from professor;
#복합 뷰
#2. Accounting 부서에 근무하는 사원에 대해 last_name,직무id,email,부서번호,부서이름 열을 포함하고 last_name순으로 출력되는 employee_deptname뷰를 생성
select * from employees;
select * from departments;
create or replace view employee_deptname
as select e.last_name,e.job_id,e.email,e.department_id,d.department_name
from employees e,departments d
where e.department_id=d.department_id and department_name='Accounting';
select * from employee_deptname;
#,뷰 생성-함수를 사용
#3. 사원 테이블에서 부서별 평균 급여로 정의되는 v_emp_avg_sal 뷰를 생성
select * from emp;
select * from dept;

select dname,avg_sal
from( select deptno,avg(sal) avg_sal
      from emp
      group by deptno) e, dept d
      where e.deptno=d.deptno;
    
create or replace view v_emp_avg_sal
as select d.dname,round(avg(sal),0) avg_sal
from emp e,dept d
where e.deptno=d.deptno
group by d.dname;

#1. test라는 테이블 스페이스를 기본 150로 생성
create tablespace test
datafile 'C:\oraclexe\test.dbf' size 150m;
#2. test/test123# 이라는 유저 생성후 디폴트 테이블스페이스는 test,temporary 테이블 스페이스는 temp을 지정
create user test identified by test123#
default tablesapce test temporary tablespace temp;
#3. member 테이블을 아래와 같이 생성 
create table member(
id number(7) not null,
name varchar2(10) not null,
regedit date,
phone number(11,2),
sex varchar(6));
#4. 생성한 member테이블에 데이터3건을 입력
insert into member(id,name,regedit,phone,sex) values(1000,'brave',sysdate,01011112222,'남');
insert into member(id,name,regedit,phone,sex) values(1001,'jerry',sysdate,010-5421-2231,'여');
insert into member(id,name,regedit,phone,sex) values(1002,'json',sysdate,010-2218-7877,'남');
#5. 생성한 test 유저가 hr의 student테이블을 select 하도록 권한 부여
grant comnnect, resource to test;

grant select on hr.member to test;
#6. 5에서 부여한 권환을 회수.
revoke select on member from test;

#롤 생성
create role hr_clerk;
create role hr_mgr
identified by manager;
#롤에 권한 부여
grant create session to hr_mgr;
grant select,insert,delete on student to hr_clerk;
#롤 부여
grant hr_clerk to hr_mgr; --롤을 롤한테
grant hr_clerk to tiger; --롤을 사용자에게
#롤 조회
select * from role_sys_privs; --롤에 부여한 시스템권한 조회
select * from role_tab_privs; --롤에 부여한 객체 권한 조회
select * from user_role_privs; --사용자가 부여받은 롤 조회
#전용 동의어 생성
connect system/manager
create table project(
    project_id number(5) constraint pro_id_pk primary key,
    project_name varchar2(100),
    studno number(5),
    profno number(5));
insert into project values(12345,'portfolio',10101,9901);

grant select on project to hr;
conn hr/hr
select * from project;
select * from system.project; --앞에 system. 을 붙여야 나옴
create synonym my_project for system.project -- 동의어 생성
select * from my_project;
#공용 동의어 생성
conn system/manager
create public synonym pub_project for project; --동의어 생성
conn hr/hr
select * from pub_project; --바로 출력
#동의어 삭제
drop synonym my_project; --private 동의어 삭제
drop public synonym pub_project; --public 동의어 삭제

#계층적 질의문--top down 방식
select deptno,dname,college
from department
start with deptno=10    --시작 데이터
connect by prior deptno=college;
--bottom up 방식
select deptno,dname,college
from department
start with deptno=102
connect by prior college=deptno;
--레벨별 구분
select lpad('    ',(level-1)*2) || dname 조직도
from department
start with dname='공과대학'
connect by prior deptno=college;

#계층구조에서 가지 제거방법
select deptno,college,dname,loc
from department
where dname!='정보미디어학부' --정보 미디어 학부만 제외
start with college is null
connect by prior deptno=college;

select deptno,college,dname,loc
from department
start with college is null
connect by prior deptno=college
and dname !='정보미디어학부'; --정보미디어학부 제외 and 정보미디어 학부에 속한 모든 학과를 제외

#계층적 질의문을 사용하여 사원 테이블에서 관리자사번,사번,사원명 순으로 topdown 형식의 계층 구조로출력
select * from emp;
select mgr,empno,ename
from emp
start with mgr is null
connect by prior empno=mgr;

select lpad('  ',(level-1)*2) || mgr,empno,ename
from emp
start with mgr is null
connect by prior empno=mgr;

#계층적 질의문을 사용하여 사원테이블에서 급여가 1000이상이고 관리자번호가 7698인 사원의 관리자사번,사번,사원명순으로출력
select mgr,empno,ename,sal
from emp
where sal>=1000
start with mgr=7698
connect by prior empno=mgr;
