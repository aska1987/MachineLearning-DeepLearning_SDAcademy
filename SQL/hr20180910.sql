#�ε��� ���� ��ȸ
select index_name,uniqueness
from user_indexes
where table_name='STUDENT';

select index_name,column_name
from user_ind_columns
where table_name='STUDENT';
#�ε��� ����
drop index fidx_stud_no_name;
#�ε��� �籸��
alter index stud_no_pk rebuild;

#1. EMPNO,ENAME,DEPTNO ������ �����ϴ� EMP ���̺��� ������ ���ʷ� EMPLOYEE2 ���̺� ����, �� ���̺��� ID,LAST_NAME,DEPT_ID �� �� �̸� ����
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

#2. ������ EMPLOYEE2 ���̺��� LAST_NAME �ʵ带 10->30 ���� ����
alter table employee2
modify(LAST_NAME varchar2(30));
desc employee2;
#3-1. ��� ���̺��� DEPTNO��ENAME�� �����ε����� �����϶�. ���� �ε����� �̸��� fidx_emp_no_ename(��, DEPTNO Į���� ������������ ENAME�÷��� ������������ ����
create index fidx_emp_no_ename
on emp(DEPTNO asc,ENAME desc);
#3-2. �Ʒ� ������ �ε����� �����θ� Ȯ���ϼ���
SQLPLUS��
conn hr/hr
set autotrace on 
select empno,ename,deptno
from emp
where ename='SMITH'
and deptno=20;
#3-3. �ε����� �籸���ϼ���.
alter index fidx_emp_no_ename rebuild;
#3-4. fidx_emp_no_ename �ε����� �����ϰ� �Ʒ� ������ �ε����� �����θ� Ȯ���ϼ���.
drop index fidx_emp_no_ename;
select empno,ename,deptno
from emp
where ename='SMITH'
and deptno=20;

#�����
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

#�ζ��� ��
select dname,avg_height,avg_weight
from (select deptno,avg(height) avg_height,avg(weight) avg_weight
from student
group by deptno) s,department d
where s.deptno=d.deptno;

#�� ��ȸ
select view_name,text
from user_views;

#�� ����
create or replace view v_stud_dept102
as select studno,name,deptno,grade
from student
where deptno=101;
#�� ����
drop view v_stud_dept102;
drop view v_stud_dept102_2;

#�ܼ� ��
#1. PROFNO,NAME,POSITION,DEPTNO ���� �����ϴ� PROFSSOR ���̺��� ������ ���ʷ� PROFESSOR2�並 �ۼ�
create view professor2
as select profno,name,position,deptno
from professor;
#���� ��
#2. Accounting �μ��� �ٹ��ϴ� ����� ���� last_name,����id,email,�μ���ȣ,�μ��̸� ���� �����ϰ� last_name������ ��µǴ� employee_deptname�並 ����
select * from employees;
select * from departments;
create or replace view employee_deptname
as select e.last_name,e.job_id,e.email,e.department_id,d.department_name
from employees e,departments d
where e.department_id=d.department_id and department_name='Accounting';
select * from employee_deptname;
#,�� ����-�Լ��� ���
#3. ��� ���̺��� �μ��� ��� �޿��� ���ǵǴ� v_emp_avg_sal �並 ����
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

#1. test��� ���̺� �����̽��� �⺻ 150�� ����
create tablespace test
datafile 'C:\oraclexe\test.dbf' size 150m;
#2. test/test123# �̶�� ���� ������ ����Ʈ ���̺����̽��� test,temporary ���̺� �����̽��� temp�� ����
create user test identified by test123#
default tablesapce test temporary tablespace temp;
#3. member ���̺��� �Ʒ��� ���� ���� 
create table member(
id number(7) not null,
name varchar2(10) not null,
regedit date,
phone number(11,2),
sex varchar(6));
#4. ������ member���̺� ������3���� �Է�
insert into member(id,name,regedit,phone,sex) values(1000,'brave',sysdate,01011112222,'��');
insert into member(id,name,regedit,phone,sex) values(1001,'jerry',sysdate,010-5421-2231,'��');
insert into member(id,name,regedit,phone,sex) values(1002,'json',sysdate,010-2218-7877,'��');
#5. ������ test ������ hr�� student���̺��� select �ϵ��� ���� �ο�
grant comnnect, resource to test;

grant select on hr.member to test;
#6. 5���� �ο��� ��ȯ�� ȸ��.
revoke select on member from test;

#�� ����
create role hr_clerk;
create role hr_mgr
identified by manager;
#�ѿ� ���� �ο�
grant create session to hr_mgr;
grant select,insert,delete on student to hr_clerk;
#�� �ο�
grant hr_clerk to hr_mgr; --���� ������
grant hr_clerk to tiger; --���� ����ڿ���
#�� ��ȸ
select * from role_sys_privs; --�ѿ� �ο��� �ý��۱��� ��ȸ
select * from role_tab_privs; --�ѿ� �ο��� ��ü ���� ��ȸ
select * from user_role_privs; --����ڰ� �ο����� �� ��ȸ
#���� ���Ǿ� ����
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
select * from system.project; --�տ� system. �� �ٿ��� ����
create synonym my_project for system.project -- ���Ǿ� ����
select * from my_project;
#���� ���Ǿ� ����
conn system/manager
create public synonym pub_project for project; --���Ǿ� ����
conn hr/hr
select * from pub_project; --�ٷ� ���
#���Ǿ� ����
drop synonym my_project; --private ���Ǿ� ����
drop public synonym pub_project; --public ���Ǿ� ����

#������ ���ǹ�--top down ���
select deptno,dname,college
from department
start with deptno=10    --���� ������
connect by prior deptno=college;
--bottom up ���
select deptno,dname,college
from department
start with deptno=102
connect by prior college=deptno;
--������ ����
select lpad('    ',(level-1)*2) || dname ������
from department
start with dname='��������'
connect by prior deptno=college;

#������������ ���� ���Ź��
select deptno,college,dname,loc
from department
where dname!='�����̵���к�' --���� �̵�� �кθ� ����
start with college is null
connect by prior deptno=college;

select deptno,college,dname,loc
from department
start with college is null
connect by prior deptno=college
and dname !='�����̵���к�'; --�����̵���к� ���� and �����̵�� �кο� ���� ��� �а��� ����

#������ ���ǹ��� ����Ͽ� ��� ���̺��� �����ڻ��,���,����� ������ topdown ������ ���� ���������
select * from emp;
select mgr,empno,ename
from emp
start with mgr is null
connect by prior empno=mgr;

select lpad('  ',(level-1)*2) || mgr,empno,ename
from emp
start with mgr is null
connect by prior empno=mgr;

#������ ���ǹ��� ����Ͽ� ������̺��� �޿��� 1000�̻��̰� �����ڹ�ȣ�� 7698�� ����� �����ڻ��,���,�������������
select mgr,empno,ename,sal
from emp
where sal>=1000
start with mgr=7698
connect by prior empno=mgr;
