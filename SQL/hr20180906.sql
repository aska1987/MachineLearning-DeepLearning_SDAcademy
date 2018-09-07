select * from emp;
select * from dept;
select ename,job,dname,loc
from emp join dept
using(deptno)
where loc='DALLAS';

#멀티미디어공학과학생이면서 키가 170이상인 학생인 학번,이름,키,학과이름,학과위치를 출력
select * from student;
select * from department;
select studno,name,height,dname,loc
from student join department
using(deptno)
where dname='멀티미디어학과'
and height>=170;

#3.Chen 사원에 관한 정보를 last_name,직무id,email,부서번호,부서이름을 last_name순으로 출력
select * from employees;
select * from departments;
select last_name,employee_id,email,department_id,department_name
from employees join departments
using(department_id)
where last_name='Chen';
#4. 3번에 이어서 Chen 사원에 관한 더 자세한 정보인 이름,직무id,부서이름,도시,주를 출력
select * from locations;
select e.last_name,e.employee_id,d.department_name,l.city,l.state_province
from employees e, departments d, locations l
where e.department_id=d.department_id 
and d.location_id=l.location_id
and e.last_name='Chen';

#사원 테이블에서 7788사원이 있는지 확인해보고 이름,부서번호,급여를 출력
select * from emp;
select ename,deptno,sal
from emp
where exists(select empno
            from emp
            where empno=7788)
and empno=7788;

#Blake와 같은 부서에 있는 모든 사원에 대해서 사원이름과 입사일을 출력
select * from emp
order by ename;
select ename,hiredate
from emp
where deptno = (select deptno
               from emp
               where INITCAP(ename)='Blake');
               
#평균 급여 이상을 받는 모든사원에 대해서 사원번호와 이름을 출력
select empno,ename
from emp
where sal>=(select avg(sal)
            from emp);
            
#부서번호와 급여가 보너스를 받는 어떤 사원의 부서번호와 급여에 일치하는 사원의 이름,부서번호 그리고 급여를 출력
select ename,deptno,sal
from emp
where sal=(select nvl2(comm,sal+comm,sal) sal
           from emp
           );
--정답--
select ename,deptno,sal
from emp
where (deptno,sal) in
        (select deptno,sal
         from emp
         where comm is not null);

desc student;
insert into student
values(10110,'홍길동','hong','1','8501011143098','85/01/01','041)630-3114',170,70,101,9903);
select studno,name
from student
where studno=10110;
select * from student;
commit;

desc department;
insert into department(deptno,dname)
values(300,'생명공학부');
select *
from department
where deptno=300;

insert into department
values(301,'환경보건학과','','');
select *
from department
where deptno=301;

insert into professor(profno,name,position,hiredate,deptno)
values(9920,'최윤식','조교수',to_date('2006/01/01','YYYY/MM/DD'),102);
select *
from professor
where profno=9920;

insert into professor
values(9910,'백미선','white','전임강사',200,sysdate,10,101);
select *
from professor
where profno=9910;

insert into professor
values(2000,'아무나','blue','부교수',250,sysdate,10,101);
insert into professor
values(2001,'갑을병','black','교수',300,sysdate,10,102);
insert into professor
values(2002,'도서관','red','전임강사',210,sysdate,10,103);
select *
from professor;

create table t_student
as select * from student
where 1=0;
insert into t_student
select * from student;
commit;

create table height_info(
studno number(5),
name varchar2(10),
height number(5,2));
create table weight_info(
studno number(5),
name varchar2(10),
weight number(5,2));

insert all
into height_info values(studno, name,height)
into weight_info values(studno, name,weight)
select studno,name,height,weight
from student
where grade >='2';
commit;
select *
from height_info;

#professor 테이블을 기준으로 테이블2개를 생성(professor_sal,professor_dept) 101번 102번 학과 교수님들의 정보를 입력
select * from professor;
1)professor_sal 테이블  profno,name,sal,comm
create table professor_sal
as select profno,name,sal,comm  from professor
where 1=0;
insert all
into professor_sal
select profno,name,sal,comm
from professor
where deptno in (101,102);
select * from professor_sal;
2)professor_dept 테이블 profno,name,deptno
create table professor_dept
as select profno,name,deptno from professor
where 1=0;
insert all
into professor_dept
select profno,name,deptno
from professor
where deptno in (101,102);
select * from professor_dept;

delete from height_info;
delete from weight_info;
commit;

insert all
when height >170 then
    into height_info values(studno,name,height)
when weight >70 then
    into weight_info values(studno,name,weight)
select studno,name,height,weight
from student
where grade >='2';

delete from height_info;
delete from weight_info;
commit;

insert first
when height >170 then
    into height_info values(studno,name,height)
when weight >70 then
    into weight_info values(studno,name,weight)
select studno,name,height,weight
from student
where grade >='2';
select * from height_info;
select * from weight_info;

create table sales(
sales_no number(4),
week_no number(2),
sales_mon number(7,2),
sales_tue number(7,2),
sales_wed number(7,2),
sales_thu number(7,2),
sales_fri number(7,2));
insert into sales values(1101,4,100,150,80,60,120);
insert into sales values(1102,5,300,300,230,120,150);
create table sales_data(
sale_no number(4),
week_no number(2),
day_no number(2),
sales number(7,2));
insert all
into sales_data values(sales_no,week_no,'1',sales_mon)
into sales_data values(sales_no,week_no,'2',sales_tue)
into sales_data values(sales_no,week_no,'3',sales_wed)
into sales_data values(sales_no,week_no,'4',sales_thu)
into sales_data values(sales_no,week_no,'5',sales_fri)
select sales_no,week_no,sales_mon,sales_tue,sales_wed,sales_thu,sales_fri
from sales;

select * from sales;
select * from sales_data
order by sale_no;

select profno,name,position
from professor
where profno = 9903;
update professor
set position='부교수'
where profno=9903;
select profno,name,position
from professor
where profno = 9903;
update professor
set sal=10000
where profno=9903;
select * from professor
where profno=9903;
#101번 학과 교수들을 모두 부교수로 변경
update professor
set position='부교수'
where deptno=101;

select studno,grade,deptno
from student
where studno=10201;
select studno,grade,deptno
from student
where studno=10103;
update student
set (grade,deptno)=(select grade,deptno
                    from student
                    where studno=10103)
where studno=10201;

delete
from student
where studno=20103;
commit;
select * from student
where studno=20103;
rollback;

delete from student 
where deptno=(select deptno
              from department
              where dname='컴퓨터공학과');
select *
from student
where deptno=(select deptno
              from department
              where dname='컴퓨터공학과');
              
create table professor_temp as
select *
from professor
where position='교수';
update professor_temp
set position='명예교수'
where position='교수';

insert into professor_temp
values(9999,'김도경','arom21','전임강사',200,sysdate,10,101);
select * from professor;
select * from professor_temp;

merge into professor p
using professor_temp f
on (p.profno=f.profno)
when matched then
update set p.position=f.position
when not matched then
insert values(f.profno,f.name,f.userid,f.position,f.sal,f.hiredate,f.comm,f.deptno);
select * from professor;

create sequence s_seq
increment by 1
start with 1
maxvalue 100;

select min_value,max_value,increment_by,last_number
from user_sequences
where sequence_name='S_SEQ';

#8000부터 시작해서 1씩 증가하고 maxvalue 10000인 ss_seq 시퀀스 생성
create sequence ss_seq
increment by 1
start with 8000
maxvalue 10000;
select min_value,max_value,increment_by,last_number
from user_sequences
where sequence_name='SS_SEQ';


#ss_seq 시퀀스를 이용하여 사원테이블에 3명을 입력
insert into emp values
(ss_seq.nextval,'cathy1','SALESMAN',7698,sysdate,800,null,20);
insert into emp values
(ss_seq.nextval,'cathy2','SALESMAN',7698,sysdate,800,null,20);
insert into emp values
(ss_seq.nextval,'cathy3','SALESMAN',7698,sysdate,800,null,20);
select * from emp;

alter sequence s_seq maxvalue 200;
select min_value,max_value,increment_by,last_number
from user_sequences
where sequence_name='S_SEQ';

drop sequence s_seq;

#1. Raphaely 보다 급여가 많은 사람이 이름과 급여를 출력하는데, 급여에 대해서는 오름차순 정렬
select * from employees;
select last_name,salary
from employees
where salary >(select salary
            from employees
            where last_name='Raphaely')
order by salary desc;
#2. 총 급여가 $35,000 가 넘는 각 직무에 대해 직무 id와 월급 총액을 출력하는데 급여 총액에 대해 오름차순으로 정렬
select * from departments;
select job_id,sum(salary)
from employees
group by job_id
having sum(salary)>35000
order by sum(salary);
#3. dept테이블의 기본 키 별로 사용되기 위한 시퀀스를 생성하세요. 시퀀스는 60에서 시작, 최대값은 200,10씩 증가하는시퀀스 (이름은 dept_id_seq)
create sequence DEPT_ID_SEQ
increment by 10
start with 60
maxvalue 200;
#4. 3에서 생성한 시퀀스의 NEXTBAL를 이용해서 DEPT테이블에 MIS,MANAGEMENT 부서를 추가해보세요.
select * from dept;
insert into DEPT 
values(DEPT_ID_SEQ.nextval,'MIS',null);
insert into DEPT 
values(DEPT_ID_SEQ.nextval,'MANAGEMENT',null);
#5. DEPT 테이블의 60번 부서명을 RESEARCH로 변경하세요.
update DEPT SET DNAME='RESEARCH'
where deptno=60;
